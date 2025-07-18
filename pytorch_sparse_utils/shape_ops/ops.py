from typing import Optional
import torch
from torch import Tensor

from ..indexing.script_funcs import (
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
    support sparse tensors.
    Sparse tensors may be reshaped in one or both of their sparse or dense dimensions.
    The intention is to be as close to feature compatibility to the built-in dense
    Tensor.reshape() as possible. As such, features like inferred dimensions (by passing
    -1 as one of the new dimensions), shape compatibility checking, etc., are supported.

    Sparse and/or dense dimensions of a sparse tensor may be reshaped independently by
    passing in the new shape(s). Reshaping the sparse dimension also updates the
    .indices() of the nonzero values of the sparse tensor. If one of the

    Args:
        tensor (Tensor): Input sparse tensor to be reshaped.
        new_sparse_shape (Optional[list[int]], optional): The new shape of the sparse
            tensor's sparse dimensions. If None, the sparse dimensions are not reshaped.
            Defaults to None.
        new_dense_shape (Optional[list[int]], optional): The new shape of the sparse
            tensor's dense dimensions. If None, the dense dimensions are not reshaped.
            Defaults to None.

    Raises:
        ValueError: If the input tensor is not sparse, or if neither of new_sparse_shape
            nor new_dense_shape are provided.
        RuntimeError: If the new shapes are incompatible, with similar conditions to
            vanilla dense reshape.

    Returns:
        Tensor: The reshaped tensor.
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


def sparse_squeeze(tensor: Tensor, dim: int) -> Tensor:
    """Squeeze (remove) a dimension of size 1 from a COO sparse tensor.

    The dimension to squeeze may be either:

        - a sparse dimension (0, ..., tensor.sparse_dim() - 1)
        - a dense dimension (tensor.sparse_dim(), ..., tensor.ndim - 1)

    If the dimension is not of length 1, the tensor is returned unchanged, as with
    the squeeze(dim) method of dense tensors.

    Args:
        tensor (Tensor): Sparse COO Tensor.
        dim (int): Dimension to squeeze.

    Returns:
        Tensor: Input tensor with specified sparse dim squeezed out, if applicable.

    Notes:
        - If the input tensor requires gradients, it must be coalesce()d before being
            passed to this function. Tensors that do not require gradients may be
            passed in un-coalesced form.
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
        tuple(new_shape),
        requires_grad=tensor.requires_grad,
        is_coalesced=tensor.is_coalesced(),
    )


def sparse_resize(tensor: Tensor, new_shape: list[int]) -> Tensor:
    """Copies the indices and values of `tensor` to a new sparse tensor
    of different shape and same number of dims"""
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
    return torch.sparse_coo_tensor(
        tensor.indices(), tensor.values(), new_shape, is_coalesced=tensor.is_coalesced()
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
    """Flattens any number of dimensions of an n-D sparse tensor"""
    assert tensor.is_sparse
    if start_axis < 0:
        start_axis = tensor.ndim + start_axis
    if end_axis < 0:
        end_axis = tensor.ndim + end_axis
    assert end_axis > start_axis
    assert start_axis >= 0
    assert end_axis <= tensor.ndim
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
