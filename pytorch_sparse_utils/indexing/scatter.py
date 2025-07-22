import torch
from torch import Tensor

from .utils import (
    _make_linear_offsets,
    get_sparse_index_mapping,
    unflatten_nd_indices,
)


# @torch.jit.script
def _merge_sorted(
    old_nd: Tensor,
    new_nd: Tensor,
    old_values: Tensor,
    new_values: Tensor,
    insertion_positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Merges two sorted sequences of sparse indices/values and return them in coalesced
    order.

    All input args must be on the same device.

    Args:
        old_nd (Tensor): [S x n_old] tensor of N-D indices
        new_nd (Tensor): [S x n_new] tensor of N-D indices
        old_values (Tensor): [n_old, ...] tensor of values
        new_values (Tensor): [n_new, ...] tensor of values
        insertion_positions (Tensor): [n_new] tensor of insertion positions in
            old_linear for each element in new_linear

    Returns:
        merged_nd: [S, n_old+n_new]
        merged_values: [n_old+n_new, ...]
    """
    device = old_nd.device
    n_old, n_new = old_nd.size(1), new_nd.size(1)
    n_total = n_old + n_new

    # determine final positions of new values
    # account for previous insertions to get final positions of new rows
    new_positions = insertion_positions + torch.arange(
        n_new, device=device, dtype=insertion_positions.dtype
    )

    # determine final positions of old values by counting how many new values are
    # inserted before each old value
    hist = torch.bincount(insertion_positions, minlength=n_old + 1)
    old_shift = torch.cumsum(hist[:-1], 0)
    old_positions = torch.arange(n_old, device=device) + old_shift

    # allocate output tensors
    merged_nd = old_nd.new_empty(old_nd.size(0), n_total)
    merged_values = old_values.new_empty((n_total,) + old_values.shape[1:])

    # insert values
    merged_nd[:, old_positions] = old_nd
    merged_nd[:, new_positions] = new_nd
    merged_values[old_positions] = old_values
    merged_values[new_positions] = new_values

    return merged_nd, merged_values


# @torch.jit.script
def scatter_to_sparse_tensor(
    sparse_tensor: Tensor,
    index_tensor: Tensor,
    values: Tensor,
    check_all_specified: bool = False,
) -> Tensor:
    """Batch updating of elements in a torch sparse tensor. Should be
    equivalent to sparse_tensor[index_tensor] = values. It works by flattening
    the sparse tensor's sparse dims and the index tensor to 1D (and converting
    n-d indices to raveled indices), then using index_copy along the flattened
    sparse tensor.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension
            [s0, s1, s2, ..., d0, d1, d2, ...]; where s0, s1, ... are
            S leading sparse dimensions and d0, d1, d2, ... are D dense dimensions.
        index_tensor (Tensor): Long tensor of dimension [b0, b1, b2, ..., S]; where
            b0, b1, b2, ... are B leading batch dimensions.
        values (Tensor): Tensor of dimension [b0, b1, b2, ... d0, d1, d2, ...], where
            dimensions are as above.
        check_all_specified (bool): If True, this function will throw a ValueError
            if any of the indices specified in index_tensor are not already present
            in sparse_tensor. Default: False.

    Returns:
        Tensor: sparse_tensor with the new values scattered into it

    Notes:
        This function uses index_copy as the underlying mechanism to write new values,
            so duplicate indices in index_tensor will have the same result as other
            uses of index_copy, i.e., the result will depend on which copy occurs last.
            This imitates the behavior of scatter-like operations rather than the
            typical coalescing deduplication behavior of sparse tensors.

    Examples:
        >>> # Create a sparse tensor with values
        >>> indices = torch.tensor([[0, 1, 2], [0, 1, 0]])
        >>> values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (3, 3, 2))

        >>> # Update existing values
        >>> update_indices = torch.tensor([[0, 0], [1, 1]])
        >>> new_values = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        >>> updated = scatter_to_sparse_tensor(sparse, update_indices, new_values)
        >>> updated.to_dense()
        tensor([[[10., 20.],  # Updated
                 [ 0.,  0.],
                 [ 0.,  0.]],
                [[ 0.,  0.],
                 [30., 40.],  # Updated
                 [ 0.,  0.]],
                [[5.,  6.],   # Unchanged
                 [ 0.,  0.],
                 [ 0.,  0.]]])

        >>> # Add new values (scatter to unspecified locations)
        >>> new_indices = torch.tensor([[0, 2], [2, 2]])
        >>> new_values = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
        >>> updated = scatter_to_sparse_tensor(sparse, new_indices, new_values)
        >>> updated.to_dense()[0, 2]  # New value added
        tensor([100., 200.])

        >>> # Batch update with multiple indices
        >>> batch_indices = torch.tensor([[[0, 0], [1, 1]],
        ...                               [[2, 0], [0, 1]]])
        >>> batch_values = torch.tensor([[[11., 12.], [13., 14.]],
        ...                              [[15., 16.], [17., 18.]]])
        >>> # Flatten batch dimensions
        >>> flat_indices = batch_indices.reshape(-1, 2)
        >>> flat_values = batch_values.reshape(-1, 2)
        >>> updated = scatter_to_sparse_tensor(sparse, flat_indices, flat_values)

        >>> # check_all_specified example
        >>> indices = torch.tensor([[0, 0], [1, 1]])
        >>> values = torch.tensor([1.0, 2.0])
        >>> sparse = torch.sparse_coo_tensor(indices.T, values, (2, 2))
        >>>
        >>> # This will succeed (all indices exist)
        >>> update_indices = torch.tensor([[0, 0]])
        >>> update_values = torch.tensor([10.0])
        >>> result = scatter_to_sparse_tensor(sparse, update_indices, update_values,
        ...                                   check_all_specified=True)
        >>>
        >>> # This will raise ValueError (index [1, 0] doesn't exist)
        >>> try:
        ...     bad_indices = torch.tensor([[1, 0]])
        ...     bad_values = torch.tensor([20.0])
        ...     result = scatter_to_sparse_tensor(sparse, bad_indices, bad_values,
        ...                                       check_all_specified=True)
        ... except ValueError as e:
        ...     print("Error:", e)
        Error: `check_all_specified` was set to True but not all gathered values were specified
    """
    if index_tensor.is_nested:
        assert values.is_nested
        index_tensor = torch.cat(index_tensor.unbind())
        values = torch.cat(values.unbind())

    dense_dim = sparse_tensor.dense_dim()
    sparse_dim = sparse_tensor.sparse_dim()
    values_batch_dims = values.shape[:-dense_dim] if dense_dim else values.shape
    if index_tensor.shape[:-1] != values_batch_dims:
        raise ValueError(
            "Expected matching batch dims for `index_tensor` and `values`, but got "
            f"batch dims {index_tensor.shape[:-1]} and "
            f"{values_batch_dims}, respectively."
        )

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor, sanitize_linear_index_tensor=False
    )

    all_specified = torch.all(is_specified_mask)

    if check_all_specified and not all_specified:
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # In-place update of existing values
    if not sparse_tensor_values.requires_grad:
        updated_values = sparse_tensor_values.index_copy_(
            0, index_search[is_specified_mask], values[is_specified_mask]
        )
    else:
        updated_values = sparse_tensor_values.index_copy(
            0, index_search[is_specified_mask], values[is_specified_mask]
        )

    if all_specified:  # No new values to append: tensor is fully updated
        return torch.sparse_coo_tensor(
            sparse_tensor.indices(),
            updated_values,
            sparse_tensor.shape,
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device,
            is_coalesced=True,
        )

    # Need to append at least one new value: pre-sort the index tensor to save
    # the final coalesce operation

    # Pull out new values and indices to be added
    new_insert_pos: Tensor = index_search[~is_specified_mask]
    new_nd_indices = index_tensor[~is_specified_mask]
    new_values = values[~is_specified_mask]

    # Get sparse shape info for linearization
    sparse_sizes = torch.tensor(
        sparse_tensor.shape[:sparse_dim], device=sparse_tensor.device
    )

    if (new_nd_indices >= sparse_sizes.unsqueeze(0)).any():
        raise ValueError(
            "`index_tensor` has indices that are out of bounds of the original "
            f"sparse tensor's sparse shape ({sparse_sizes})."
        )

    # Obtain linearized versions of all indices for sorting
    old_indices_nd = sparse_tensor.indices()
    linear_offsets = _make_linear_offsets(sparse_sizes)
    new_indices_lin: Tensor = (new_nd_indices * linear_offsets).sum(-1)

    # Find duplicate linear indices
    unique_new_indices_lin, inverse = torch.unique(
        new_indices_lin, sorted=True, return_inverse=True
    )

    # Use inverse of indices unique to write to new values tensor and tensor of
    # insertion positions
    deduped_new_values = new_values.new_empty(
        (unique_new_indices_lin.size(0),) + new_values.shape[1:]
    )
    deduped_insert_pos = new_insert_pos.new_empty(unique_new_indices_lin.size(0))
    # Deciding which duplicate value wins is offloaded to index_copy
    deduped_new_values.index_copy_(0, inverse, new_values)
    deduped_insert_pos.index_copy_(0, inverse, new_insert_pos)

    # Convert uniqueified flattened indices to n-D for inclusion in indices tensor
    unique_new_indices_nd = unflatten_nd_indices(
        unique_new_indices_lin.unsqueeze(0), sparse_sizes, linear_offsets
    )

    # Concatenate old and new indices/values and sort
    combined_indices, combined_values = _merge_sorted(
        old_indices_nd,
        unique_new_indices_nd,
        updated_values,
        deduped_new_values,
        deduped_insert_pos,
    )

    return torch.sparse_coo_tensor(
        combined_indices,
        combined_values,
        sparse_tensor.shape,
        dtype=sparse_tensor.dtype,
        device=sparse_tensor.device,
        is_coalesced=True,
    )
