import torch
from torch import Tensor

from .utils import (
    gather_mask_and_fill,
    get_sparse_index_mapping,
)


@torch.jit.script
def sparse_select(tensor: Tensor, axis: int, index: int) -> Tensor:
    """Selects a single subtensor from a sparse tensor along the specified axis.

    This function extracts a slice of a sparse tensor by selecting elements
    where the coordinate along the specified sparse axis matches the given index value.
    Unlike Pytorch's built-in indexing, this implementation properly handles
    gradient flow for sparse tensors.

    Args:
        tensor (Tensor): The input sparse tensor.
        axis (int): The dimension along which to select the subtensor. Negative axes
            are supported.
        index (int): The index to select along the specified axis. Negative indices
            are supported.

    Returns:
        Tensor: A new sparse tensor with one fewer dimension than the input tensor.
            The shape will be tensor.shape[:axis] + tensor.shape[axis+1:].

    Raises:
        ValueError: If the input tensor is not sparse, or if the axis or index are
            out of bounds.

    Examples:
        Create a sparse tensor and select slices:

        >>> # Create a sparse tensor with dimensions (3, 4, 2), where the last
        >>> # dimension is a dense dimension
        >>> i = torch.tensor([[0, 1], [1, 3], [2, 2]].T)
        >>> v = torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        >>> x = torch.sparse_coo_tensor(i, v, (3, 4, 2))
        >>>
        >>> # Select values along sparse dimension 0
        >>> slice_0 = sparse_select(x, 0, 1)  # Get elements where dim 0 == 1
        >>> # slice_0 has shape (4, 2)
        >>>
        >>> # Select values along dense dimension 2
        >>> slice_2 = sparse_select(x, 2, 0)  # Get elements where dim 2 == 0
        >>> # slice_2 has shape (3, 4)
    """
    if not tensor.is_sparse:
        raise ValueError("Input tensor is not sparse.")

    if not isinstance(index, int):
        raise ValueError(f"Expected integer index, got type {type(index)}")

    # Normalize negative axis
    orig_axis = axis
    if axis < 0:
        axis = tensor.ndim + axis

    # Validate axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {orig_axis} out of bounds for tensor with {tensor.ndim} dimensions"
        )

    # Normalize negative index
    orig_index = index
    if index < 0:
        index = tensor.size(axis) + index

    # check if index is in bounds
    if index < 0 or index >= tensor.size(axis):
        # weird string construction to work around torchscript not liking concatting
        # multiple f strings
        error_str = "Index " + str(orig_index) + " is out of bounds on axis "
        error_str += (
            str(orig_axis) + " for tensor with shape " + str(tensor.shape) + "."
        )
        raise ValueError(error_str)

    tensor = tensor.coalesce()

    sparse_dims = tensor.sparse_dim()
    if axis < sparse_dims:
        # Selection along a sparse dimension
        index_mask = tensor.indices()[axis] == index
        values = tensor.values()[index_mask]
        indices = torch.cat(
            [
                tensor.indices()[:axis, index_mask],
                tensor.indices()[axis + 1 :, index_mask],
            ]
        )
    else:
        # Selecting along a dense dimension
        # This means we just index the values tensor along the appropriate dim
        # and the sparse indices stay the same
        indices = tensor.indices()
        dense_axis = axis - sparse_dims + 1  # +1 because first dim of values is nnz
        values = tensor.values().select(dense_axis, index)

    return torch.sparse_coo_tensor(
        indices, values, tensor.shape[:axis] + tensor.shape[axis + 1 :]
    ).coalesce()


@torch.jit.script
def _sparse_index_select_inner(
    tensor_indices: Tensor, tensor_values: Tensor, axis: int, index: Tensor
) -> tuple[Tensor, Tensor]:
    """Inner implementation of sparse_index_select.

    This function performs the actual selection of values from a sparse tensor's
    internal representation. It constructs masks to identify which elements match the
    requested indices, then builds new indices and values tensors for the result.

    Args:
        tensor_indices (Tensor): The indices tensor from a sparse COO tensor
            (shape [sparse_dims, nnz]).
        tensor_values (Tensor): The values tensor from a sparse COO tensor
            (shape [nnz, ...]).
        axis (int): The dimension along which to select values. Negative axes are
            not supported.
        index (Tensor): The indices of the values to select.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - new_indices (Tensor): The indices for the new sparse tensor.
            - new_values (Tensor): The values for the new sparse tensor.

    Raises:
        ValueError: If axis is negative or out of bounds.
    """
    sparse_dims = tensor_indices.size(0)
    dense_dims = tensor_values.ndim - 1
    if axis < 0:
        raise ValueError(
            "`sparse_index_select_inner` does not support negative axes; got "
            f"{axis}. Please normalize axis or use `sparse_index_select`"
        )
    elif axis > sparse_dims + dense_dims:
        # weird construction required by torchscript when concatting multiple format strings
        error_msg = ["axis", str(axis), "is out of bounds for sparse tensor with"]
        error_msg += [
            str(sparse_dims),
            "sparse dims and",
            str(dense_dims),
            "dense dims.",
        ]
        # error_msg = "axis " + str(axis) + " is out of bounds for sparse tensor "
        # error_msg += "with " + str(sparse_dims) + " sparse dims and " + str(dense_dims)
        # error_msg += " dense dims."
        raise ValueError(" ".join(error_msg))

    if axis < sparse_dims:
        # Selection along a sparse dimension
        # Create masks for each index in the index tensor
        index_masks = tensor_indices[axis].unsqueeze(0) == index.unsqueeze(1)

        # Get indices where matches occur
        indices_where = torch.where(index_masks)
        selected_items = indices_where[1]

        # Extract matched values and indices
        new_values = tensor_values[selected_items]
        selected_indices = tensor_indices[:, selected_items]

        # Create new indices tensor with proper structure
        leading_indices = selected_indices[:axis]
        # Map matched positions to their corresponding index tensor positions
        axis_indices = indices_where[0].unsqueeze(0)
        trailing_indices = selected_indices[axis + 1 :]

        new_indices = torch.cat([leading_indices, axis_indices, trailing_indices], 0)
        return new_indices, new_values
    else:
        # Selection is along a dense dimension
        # We just select the appropriate values from the value tensor
        dense_dim = axis - sparse_dims + 1  # +1 because first dim of values is nnz
        new_values = torch.index_select(tensor_values, dense_dim, index)

        return tensor_indices, new_values


@torch.jit.script
def sparse_index_select(
    tensor: Tensor,
    axis: int,
    index: Tensor,
    check_bounds: bool = True,
    disable_builtin_fallback: bool = False,
) -> Tensor:
    """Selects values from a sparse tensor along a specified dimension.

    This function is equivalent to tensor.index_select(axis, index) but works
    correctly with the backward pass for sparse tensors. It returns a new sparse
    tensor containing only the values at the specified indices along the given axis.

    This function falls back to the built-in tensor.index_select(axis, index)
    when gradients are not required. Benchmarking seems to indicate the built-in
    version is generally faster and more memory efficient except for some specialized
    situations on CUDA. You can always use the custom implementation by setting this
    function's input argument disable_builtin_fallback to True.

    Note that the built-in tensor.index_select will trigger mysterious errors
    of the form "RuntimeError: CUDA error: device-side assert triggered" if it is
    given indices outside the bounds of a sparse tensor.
    Unlike the built-in tensor.index_select, this function validates that indices
    are within bounds (when check_bounds=True), making it a safer alternative even
    when gradient support isn't needed.

    Args:
        tensor (Tensor): The input sparse tensor from which to select values.
        axis (int): The dimension along which to select values. Can be negative
            to index from the end.
        index (Tensor): The indices of the values to select along the specified
            dimension. Must be a 1D tensor or scalar of integer dtype.
        check_bounds (bool, optional): Whether to check if indices are within bounds.
            Set to False if indices are guaranteed to be in-bounds to avoid a CPU sync
            on CUDA tensors. Benchmarking shows the bounds check leads to an overhead
            of about 5% on cpu and 10% on cuda. Defaults to True.
        disable_builtin_fallback (bool, optional): Whether to always use the custom
            gradient-tracking version of index_select, even when gradients are not
            needed. Does nothing if the input sparse tensor does require gradients.
            Defaults to False.

    Returns:
        Tensor: A new sparse tensor containing the selected values.

    Raises:
        ValueError:
            - If the input tensor is not sparse.
            - If the index tensor has invalid shape or is not an integer tensor.
            - If the axis is out of bounds for tensor dimensions.
            - If check_bounds is True and the index tensor contains out-of-bounds
              indices.
    """
    if not tensor.is_sparse:
        raise ValueError("Input tensor must be sparse")

    # Validate index tensor
    if torch.is_floating_point(index):
        raise ValueError(f"Received index tensor of non-integer dtype: {index.dtype}")
    if index.ndim > 1:
        raise ValueError(f"Index tensor must be 0D or 1D, got {index.ndim}D")
    elif index.ndim == 0:
        index = index.unsqueeze(0)

    # Normalize negative axis
    orig_axis = axis
    if axis < 0:
        axis = tensor.ndim + axis

    # Validate axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {orig_axis} out of bounds for tensor with {tensor.ndim} dimensions"
        )

    # Validate index bounds (optional)
    if check_bounds and index.numel() > 0:
        out_of_bounds = ((index < 0) | (index >= tensor.shape[axis])).any()
        if out_of_bounds:  # cpu sync happens here
            raise ValueError(
                f"Index tensor has entries out of bounds for axis {orig_axis} with size {tensor.shape[axis]}"
            )

    if not tensor.requires_grad and not disable_builtin_fallback:
        # Fall back to built-in implementation
        return tensor.index_select(axis, index.long()).coalesce()

    tensor = tensor.coalesce()

    tensor_indices = tensor.indices()
    tensor_values = tensor.values()

    new_indices, new_values = _sparse_index_select_inner(
        tensor_indices, tensor_values, axis, index
    )

    new_shape = list(tensor.shape)
    new_shape[axis] = len(index)

    return torch.sparse_coo_tensor(new_indices.long(), new_values, new_shape).coalesce()


@torch.jit.script
def batch_sparse_index(
    sparse_tensor: Tensor, index_tensor: Tensor, check_all_specified: bool = False
) -> tuple[Tensor, Tensor]:
    """Batch selection of elements from a torch sparse tensor. The index tensor may
    have arbitrary batch dimensions.

    If dense_tensor = sparse_tensor.to_sparse(), the equivalent dense indexing
    operation would be dense_tensor[index_tensor.unbind(-1)]

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension
            [S0, S1, ..., Sn-1, D0, D1, ..., Dm-1], with n sparse dimensions and
            m dense dimensions.
        index_tensor (Tensor): Long tensor of dimension [B0, B1, ..., Bp-1, n]
            with optional p leading batch dimensions and final dimension corresponding
            to the sparse dimensions of the sparse tensor. Negative indices are not
            supported and will be considered unspecified.
        check_all_specified (bool): If True, this function will raise a
            ValueError if any of the indices in `index_tensor` are not specified
            in `sparse_tensor`. If False, selections at unspecified indices will be
            returned with padding values of 0. Defaults to False.

    Returns:
        Tensor: Tensor of dimension [B0, B1, ..., Bp-1, D0, D1, ..., Dm-1].
        Tensor: Boolean tensor of dimension [B0, B1, ..., Bp-1], where each element is
            True if the corresponding index is a specified (nonzero) element of the
            sparse tensor and False if not.

    Raises:
        ValueError: If `check_all_specified` is set to True and not all indices in
            `index_tensor` had associated values specified in `sparse_tensor`, or if
            `index_tensor` is a nested tensor (feature planned but not implemented yet)
    """
    if index_tensor.is_nested:
        raise ValueError("Nested index tensor not supported yet")

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()
    dense_dim = sparse_tensor.dense_dim()

    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    selected: Tensor = gather_mask_and_fill(
        sparse_tensor_values, index_search, is_specified_mask
    )

    out_shape = index_tensor.shape[:-1]
    assert is_specified_mask.shape == out_shape
    if dense_dim > 0:
        out_shape = out_shape + (sparse_tensor.shape[-dense_dim:])

    assert selected.shape == out_shape

    return selected, is_specified_mask
