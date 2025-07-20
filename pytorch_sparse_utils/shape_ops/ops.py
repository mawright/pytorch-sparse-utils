from typing import Optional
import torch
from torch import Tensor

from ..indexing.utils import (
    flatten_sparse_indices,
    unflatten_nd_indices,
    flatten_nd_indices,
)
from .helpers import check_valid, do_infer_dim, verify_shape_numel


@torch.jit.script
def sparse_reshape(
    tensor: Tensor,
    new_sparse_shape: Optional[list[int]] = None,
    new_dense_shape: Optional[list[int]] = None,
) -> Tensor:
    """General-purpose .reshape() equivalent for sparse tensors.

    This function serves as an equivalent of the .reshape() operation, which does not
    support sparse tensors. Sparse tensors may be reshaped in one or both of their
    sparse or dense dimensions. The function supports inferred dimensions (by passing
    -1 as one of the new dimensions) and shape compatibility checking, similar to
    the built-in dense Tensor.reshape().

    Sparse and/or dense dimensions of a sparse tensor may be reshaped independently by
    passing in the new shape(s). Reshaping the sparse dimension also updates the
    .indices() of the nonzero values of the sparse tensor.

    Args:
        tensor (Tensor): Input sparse tensor to be reshaped.
        new_sparse_shape (Optional[list[int]], optional): The new shape of the sparse
            tensor's sparse dimensions. If None, the sparse dimensions are not reshaped.
            Supports -1 for dimension inference. Defaults to None.
        new_dense_shape (Optional[list[int]], optional): The new shape of the sparse
            tensor's dense dimensions. If None, the dense dimensions are not reshaped.
            Supports -1 for dimension inference. Defaults to None.

    Raises:
        ValueError: If the input tensor is not sparse, or if neither of new_sparse_shape
            nor new_dense_shape are provided.
        RuntimeError: If the new shapes are incompatible with the tensor's number
            of elements.

    Returns:
        Tensor: The reshaped tensor.

    Examples:
        >>> # Reshape sparse dimensions of a 2D sparse tensor
        >>> indices = torch.tensor([[0, 1, 2], [1, 0, 1]])
        >>> values = torch.tensor([1.0, 2.0, 3.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (3, 2))
        >>> reshaped = sparse_reshape(sparse, new_sparse_shape=[2, 3])
        >>> reshaped.shape
        torch.Size([2, 3])
        >>> reshaped.to_dense()
        tensor([[0., 1., 0.],
                [2., 0., 3.]])

        >>> # Reshape with dimension inference (-1)
        >>> sparse = torch.sparse_coo_tensor(indices, values, (3, 2))
        >>> reshaped = sparse_reshape(sparse, new_sparse_shape=[-1])
        >>> reshaped.shape
        torch.Size([6])

        >>> # Reshape dense dimensions of a hybrid sparse tensor
        >>> indices = torch.tensor([[0, 1]])
        >>> values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 3))
        >>> reshaped = sparse_reshape(sparse, new_dense_shape=[3, 1])
        >>> reshaped.shape
        torch.Size([2, 3, 1])
        >>> reshaped.values().shape
        torch.Size([2, 3, 1])

        >>> # Reshape both sparse and dense dimensions
        >>> indices = torch.tensor([[0, 1, 2, 3]])
        >>> values = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (4, 2))
        >>> reshaped = sparse_reshape(sparse,
        ...                          new_sparse_shape=[2, 2],
        ...                          new_dense_shape=[2, 1])
        >>> reshaped.shape
        torch.Size([2, 2, 2, 1])
    """
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")

    if new_sparse_shape is None and new_dense_shape is None:
        raise ValueError(
            "Expected one or both of new_sparse_shape and new_dense_shape but got neither."
        )

    in_sparse_shape = list(tensor.shape[: tensor.sparse_dim()])
    in_dense_shape = list(tensor.shape[tensor.sparse_dim() :])

    # Input checks for new sparse shape
    if new_sparse_shape is not None:
        check_valid(new_sparse_shape, "sparse")
        new_sparse_shape = list(new_sparse_shape)
        new_sparse_shape = do_infer_dim(new_sparse_shape, in_sparse_shape, "sparse")
        verify_shape_numel(new_sparse_shape, in_sparse_shape, "sparse")

    # Input checks for new dense shape
    if new_dense_shape is not None:
        check_valid(new_dense_shape, "dense")
        new_dense_shape = list(new_dense_shape)
        new_dense_shape = do_infer_dim(new_dense_shape, in_dense_shape, "dense")
        verify_shape_numel(new_dense_shape, in_dense_shape, "dense")

    tensor = tensor.coalesce()  # no-op if already coalesced

    # Compute new sparse indices
    if new_sparse_shape is not None:
        indices = tensor.indices()
        flat_indices, _ = flatten_nd_indices(
            indices, torch.tensor(in_sparse_shape, device=tensor.device)
        )
        new_indices = unflatten_nd_indices(
            flat_indices, torch.tensor(new_sparse_shape, device=tensor.device)
        )
        new_shape = list(new_sparse_shape)
    else:
        new_indices = tensor.indices()
        new_shape = list(in_sparse_shape)

    # Reshape sparse values
    if new_dense_shape is not None:
        values = tensor.values()
        nnz = values.size(0)
        new_values = values.reshape([nnz] + list(new_dense_shape))
        new_shape += list(new_dense_shape)
    else:
        new_values = tensor.values()
        new_shape += list(in_dense_shape)

    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        new_shape,
        is_coalesced=tensor.is_coalesced(),  # index order unchanged
    ).coalesce()


@torch.jit.script
def sparse_squeeze(tensor: Tensor, dim: int) -> Tensor:
    """Squeeze (remove) a dimension of size 1 from a COO sparse tensor.

    The dimension to squeeze may be either a sparse dimension (0 to tensor.sparse_dim()-1)
    or a dense dimension (tensor.sparse_dim() to tensor.ndim-1). If the specified
    dimension is not of size 1, the tensor is returned unchanged, consistent with
    the behavior of squeeze() for dense tensors.

    Args:
        tensor (Tensor): Sparse COO tensor to squeeze.
        dim (int): Dimension to squeeze. Supports negative indexing.

    Returns:
        Tensor: Input tensor with specified dimension squeezed out if it has size 1,
            otherwise returns the input tensor unchanged.

    Raises:
        ValueError: If the input is not a sparse tensor, or if a gradient-requiring
            tensor is passed without being coalesced.
        IndexError: If dim is out of range for the tensor.

    Notes:
        - If the input tensor requires gradients, it must be coalesce()d before being
          passed to this function. Tensors that do not require gradients may be
          passed in un-coalesced form.

    Examples:
        >>> # Squeeze a sparse dimension
        >>> indices = torch.tensor([[0, 0], [0, 1]])
        >>> values = torch.tensor([1.0, 2.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (1, 2))
        >>> squeezed = sparse_squeeze(sparse, dim=0)
        >>> squeezed.shape
        torch.Size([2])

        >>> # Squeeze a dense dimension in hybrid sparse tensor
        >>> indices = torch.tensor([[0, 1]])
        >>> values = torch.tensor([[[1.0]], [[2.0]]])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 1, 1))
        >>> squeezed = sparse_squeeze(sparse, dim=2)  # Squeeze last (dense) dim
        >>> squeezed.shape
        torch.Size([2, 1])
        >>> squeezed.values().shape
        torch.Size([2, 1])

        >>> # Dimension not squeezable (size != 1)
        >>> indices = torch.tensor([[0, 1], [0, 1]])
        >>> values = torch.tensor([1.0, 2.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 2))
        >>> result = sparse_squeeze(sparse, dim=0)
        >>> result.shape  # Unchanged
        torch.Size([2, 2])

        >>> # Negative indexing
        >>> indices = torch.tensor([[0, 0], [0, 1]])
        >>> values = torch.tensor([1.0, 2.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (1, 2))
        >>> squeezed = sparse_squeeze(sparse, dim=-2)  # Same as dim=0
        >>> squeezed.shape
        torch.Size([2])
    """
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")

    if tensor.requires_grad and not tensor.is_coalesced():
        raise ValueError(
            "Tensors that require gradients must be coalesced before being passed "
            "to sparse_squeeze."
        )

    ndim = tensor.ndim
    dim = dim if dim >= 0 else ndim + dim  # handle negative indexing
    if not 0 <= dim < ndim:
        raise IndexError(
            f"dim {dim} is out of range for tensor with {ndim} dimensions."
        )

    shape = list(tensor.shape)
    if shape[dim] != 1:
        return tensor  # unsqueezable

    sparse_dims = tensor.sparse_dim()
    if tensor.requires_grad:
        indices = tensor.indices()
        values = tensor.values()
    else:
        indices = tensor._indices()
        values = tensor._values()

    if dim < sparse_dims:  # Squeeze sparse dim
        new_indices = torch.cat((indices[:dim], indices[dim + 1 :]), 0)
        new_values = values
    else:  # Squeeze dense dim
        new_indices = indices
        new_values = values.squeeze(dim - sparse_dims + 1)

    new_shape = shape[:dim] + shape[dim + 1 :]

    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        new_shape,
        is_coalesced=tensor.is_coalesced(),
    )


@torch.jit.script
def sparse_resize(tensor: Tensor, new_shape: list[int]) -> Tensor:
    """Resize a sparse tensor by expanding its shape while preserving indices and values.

    Creates a new sparse tensor with the specified shape, copying the indices and
    values from the input tensor. The new shape must be at least as large as the
    existing shape in every dimension. This is useful as a replacement for the built-in
    Tensor.sparse_resize_(), which does not support autograd.For dense dimensions that
    are increased in size, zeros are prepended to the values, similar to the built-in
    Tensor.sparse_resize_().

    Args:
        tensor (Tensor): Input sparse COO tensor to resize.
        new_shape (list[int]): New shape for the tensor. Must have the same number
            of dimensions as the input tensor and be at least as large in each
            dimension.

    Returns:
        Tensor: A new sparse tensor with the specified shape containing the same
            nonzero values at the same indices as the input.

    Raises:
        ValueError: If the input is not sparse, if the number of dimensions differs,
            or if any dimension in new_shape is smaller than the corresponding
            dimension in the input tensor.

    Examples:
        >>> # Resize a 2D sparse tensor to a larger shape
        >>> indices = torch.tensor([[0, 1], [1, 0]])
        >>> values = torch.tensor([1.0, 2.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 2))
        >>> resized = sparse_resize(sparse, [4, 4])
        >>> resized.shape
        torch.Size([4, 4])
        >>> resized.to_dense()
        tensor([[0., 1., 0., 0.],
                [2., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]])

        >>> # Resize with dense dimension expansion
        >>> indices = torch.tensor([[0, 1]])
        >>> values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 2))
        >>> resized = sparse_resize(sparse, [2, 4])
        >>> resized.shape
        torch.Size([2, 4])
        >>> resized.values()
        tensor([[0., 0., 1., 2.],
                [0., 0., 3., 4.]])

        >>> # Resize both sparse and dense dimensions
        >>> indices = torch.tensor([[0, 1]])
        >>> values = torch.tensor([[1.0], [2.0]])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 1))
        >>> resized = sparse_resize(sparse, [4, 3])
        >>> resized.shape
        torch.Size([4, 3])
        >>> resized.values()
        tensor([[0., 0., 1.],
                [0., 0., 2.]])
    """
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")
    if not len(new_shape) == tensor.ndim:
        raise ValueError(
            f"New shape {new_shape} has different number of dims "
            f"than existing shape {tensor.shape}"
        )
    if not all(new >= old for new, old in zip(new_shape, tensor.shape)):
        raise ValueError(
            "New shape must be at least as large as existing shape in every dim, but "
            f"got new shape {new_shape} and existing shape {tensor.shape}"
        )
    tensor = tensor.coalesce()

    # Split new shape into sparse and dense parts
    sparse_dim = tensor.sparse_dim()
    new_dense_shape = new_shape[sparse_dim:]
    old_dense_shape = list(tensor.shape[sparse_dim:])

    # do padding if expansion of dense dimensions requested
    if (
        len(new_dense_shape) > 0
        and len(old_dense_shape) > 0
        and new_dense_shape != old_dense_shape
    ):
        nnz = tensor._nnz()
        values = tensor.values()
        new_values = torch.zeros(
            [nnz] + new_dense_shape, device=values.device, dtype=values.dtype
        )
        narrowed = new_values
        for dim, (old_size, new_size) in enumerate(
            zip(old_dense_shape, new_dense_shape)
        ):
            start_index = new_size - old_size
            narrowed = narrowed.narrow(dim + 1, start_index, new_size - start_index)
        narrowed.copy_(values)
    else:
        new_values = tensor.values()

    return torch.sparse_coo_tensor(
        tensor.indices(), new_values, new_shape, is_coalesced=tensor.is_coalesced()
    ).coalesce()


@torch.jit.script
def sparse_flatten_hw(tensor: Tensor) -> Tensor:
    """Flattens the middle 2 dimensions of a 4D tensor"""
    assert tensor.is_sparse
    assert tensor.ndim == 4
    tensor = tensor.coalesce()
    indices = tensor.indices()
    i = indices[1]
    j = indices[2]
    H = tensor.shape[1]
    W = tensor.shape[2]
    ij = (i * W + j).unsqueeze(0)
    new_shape = tensor.shape[:1] + (H * W,) + tensor.shape[3:]
    new_indices = torch.cat([indices[:1], ij, indices[3:]], 0).long()
    return torch.sparse_coo_tensor(new_indices, tensor.values(), new_shape).coalesce()


@torch.jit.script
def sparse_flatten(tensor: Tensor, start_axis: int, end_axis: int) -> Tensor:
    """Flattens consecutive sparse dimensions of a sparse COO tensor.

    Flattens the dimensions from start_axis to end_axis (both inclusive)
    into a single dimension. The flattened dimension is placed at the position
    of start_axis. This operation updates the indices to reflect the new
    flattened structure.

    The current version of this function cannot flatten dense dimensions, and the
    function will raise NotImplementedError if the to-be-flattened axes include dense
    dimensions.

    Args:
        tensor (Tensor): Input sparse COO tensor to flatten.
        start_axis (int): Starting dimension for flattening (inclusive).
            Supports negative indexing.
        end_axis (int): Ending dimension for flattening (inclusive).
            Supports negative indexing.

    Returns:
        Tensor: Sparse tensor with dimensions [start_axis, end_axis) flattened
            into a single dimension at position start_axis.

    Raises:
        ValueError: If tensor is not sparse, or if axis indices are invalid.
        IndexError: If start_axis or end_axis is out of range of the tensor's dims.
        NotImplementedError: If dense dimensions are requested to be flattened.

    Examples:
        >>> # Flatten a 3D sparse tensor to 2D
        >>> indices = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1]])
        >>> values = torch.tensor([1.0, 2.0, 3.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 2, 2))
        >>> flattened = sparse_flatten(sparse, start_axis=1, end_axis=2)
        >>> flattened.shape
        torch.Size([2, 4])
        >>> flattened.to_dense()
        tensor([[1., 0., 0., 0.],
                [0., 2., 0., 3.]])

        >>> # Flatten with negative indexing
        >>> flattened = sparse_flatten(sparse, start_axis=-2, end_axis=-1)
        >>> flattened.shape
        torch.Size([2, 4])

        >>> # Flatten all dimensions
        >>> flattened = sparse_flatten(sparse, start_axis=0, end_axis=-1)
        >>> flattened.shape
        torch.Size([8])
        >>> flattened.to_dense()
        tensor([1., 0., 0., 3., 0., 2., 0., 0.])
    """
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")
    min_neg = -tensor.ndim
    max_pos = tensor.ndim - 1
    if start_axis < min_neg or start_axis > max_pos:
        raise IndexError(
            f"Dimension out of range (expected to be in [{min_neg}, {max_pos}], but "
            f"got {start_axis})"
        )
    if end_axis < min_neg or end_axis > max_pos:
        raise IndexError(
            f"Dimension out of range (expected to be in [{min_neg}, {max_pos}], but "
            f"got {end_axis})"
        )

    # Normalize negative dims
    if start_axis < 0:
        start_axis = tensor.ndim + start_axis
    if end_axis < 0:
        end_axis = tensor.ndim + end_axis

    if not end_axis > start_axis:
        raise ValueError(
            "Expected end_axis to be greater than start_axis, but got "
            f"normalized axes {end_axis} and {start_axis}, respectively."
        )
    if not end_axis <= tensor.ndim:
        raise ValueError(
            "Expected end_axis to be less than number of tensor dims, but got "
            f"{end_axis} and {tensor.ndim}, respectively."
        )

    n_sparse = tensor.sparse_dim()
    if start_axis >= n_sparse or end_axis >= n_sparse:
        raise NotImplementedError(
            "sparse_flatten does not currently support flattening dense dims."
        )
    tensor = tensor.coalesce()

    new_indices, new_shape, _ = flatten_sparse_indices(tensor, start_axis, end_axis)
    assert isinstance(new_shape, Tensor)
    new_shape_list: list[int] = new_shape.tolist()
    return torch.sparse_coo_tensor(
        new_indices,
        tensor.values(),
        new_shape_list,
        is_coalesced=tensor.is_coalesced(),  # indices still unique and in correct order
    )
