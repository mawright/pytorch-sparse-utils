import torch
from torch import Tensor
from pytorch_sparse_utils.indexing.utils import flatten_nd_indices


@torch.jit.script
def unique_rows(tensor: Tensor, sorted: bool = True) -> Tensor:
    """Returns the indices of the unique rows (first dimension) of a 2D integer tensor.

    Args:
        tensor (Tensor): A 2D tensor of integer type.
        sorted (bool): Whether to sort the indices of unique rows before returning.
            If False, returned indices will be in lexicographic order of the rows.

    Returns:
        Tensor: A 1D tensor whose elements are the indices of the unique rows of
            the input tensor, i.e., if the return tensor is `inds`, then
            tensor[inds] gives a 2D tensor of all unique rows of the input tensor.

    Raises:
        OverflowError: If the tensor has values that are large enough to cause overflow
            errors when hashing each row to a single value.

    Examples:
        >>> tensor = torch.tensor([[1, 2, 3],
        ...                        [4, 5, 6],
        ...                        [1, 2, 3],  # Duplicate of row 0
        ...                        [7, 8, 9],
        ...                        [4, 5, 6]])  # Duplicate of row 1
        >>> unique_indices = unique_rows(tensor)
        >>> unique_indices
        tensor([0, 1, 3])
        >>> tensor[unique_indices]  # All unique rows
        tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    """
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got ndim={tensor.ndim}")
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError(f"Expected integer tensor, got dtype={tensor.dtype}")

    max_vals = tensor.max(0).values
    min_vals = tensor.min(0).values

    # Check for overflow problems
    INT64_MAX = 9223372036854775807
    if (max_vals >= INT64_MAX).any():
        raise OverflowError(
            f"Tensor contains values at or near maximum int64 value ({INT64_MAX}), "
            "which would lead to overflow errors when computing unique rows."
        )

    log_sum = (max_vals + 1).log().sum()
    log_max = torch.tensor(INT64_MAX, device=max_vals.device).log()

    if log_sum > log_max:
        raise OverflowError(
            "Hashing rows would cause integer overflow. Maximum hashed row product is "
            f"approx {log_sum.exp()} compared to max int64 value of {INT64_MAX}."
        )

    # Handle negative values by shifting to nonnegative
    has_negs = min_vals < 0
    if has_negs.any():
        # Shift each column to be nonnegative
        neg_shift = torch.where(has_negs, min_vals, min_vals.new_zeros([]))
        tensor = tensor - neg_shift
        max_vals = max_vals - neg_shift

    tensor_flat, _ = flatten_nd_indices(tensor.T.long(), max_vals)
    tensor_flat: Tensor = tensor_flat.squeeze(0)

    unique_flat_indices, unique_inverse = torch.unique(tensor_flat, return_inverse=True)
    unique_row_indices: Tensor = unique_inverse.new_full(
        (unique_flat_indices.size(0),), tensor_flat.size(0)
    )
    unique_row_indices.scatter_reduce_(
        0,
        unique_inverse,
        torch.arange(tensor_flat.size(0), device=tensor.device),
        "amin",
    )
    if sorted:
        unique_row_indices = unique_row_indices.sort().values
    return unique_row_indices
