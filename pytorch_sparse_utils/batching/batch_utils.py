from typing import Optional, Union
from itertools import accumulate
import operator

import torch
from torch import Tensor

from ..validation import validate_atleast_nd


@torch.jit.script
def split_batch_concatenated_tensor(
    tensor: Tensor, batch_offsets: Tensor
) -> list[Tensor]:
    """Split a batch-concatenated tensor based on batch offsets.

    Args:
        tensor (Tensor): Tensor to split with shape (total_length, D1, D2, ..., Dn)
        batch_offsets (Tensor): Tensor containing offsets where each batch starts.
            May or may not include 0 as the first element.
            May or may not include len(tensor) as the last element.

    Returns:
        list[Tensor]: List of tensors, one for each batch, where each tensor has
            shape (seq_length_i, D1, D2, ..., Dn).

    Examples:
        >>> # Split a concatenated tensor with 3 sequences
        >>> tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        >>> batch_offsets = torch.tensor([0, 2, 3, 5])
        >>> splits = split_batch_concatenated_tensor(tensor, batch_offsets)
        >>> [s.tolist() for s in splits]
        [[[1, 2], [3, 4]], [[5, 6]], [[7, 8], [9, 10]]]

        >>> # Works without leading 0 or trailing length
        >>> batch_offsets = torch.tensor([2, 3])
        >>> splits = split_batch_concatenated_tensor(tensor, batch_offsets)
        >>> [s.tolist() for s in splits]
        [[[1, 2], [3, 4]], [[5, 6]], [[7, 8], [9, 10]]]
    """
    if batch_offsets[0] == 0:
        batch_offsets = batch_offsets[1:]
    if batch_offsets[-1] == tensor.size(0):
        batch_offsets = batch_offsets[:-1]

    # cpu transfer required for tensor_split
    split_tensor = torch.tensor_split(tensor, batch_offsets.cpu())

    return list(split_tensor)


@torch.jit.script
def normalize_batch_offsets(
    batch_offsets: Union[Tensor, list[int]], total_length: int
) -> Union[Tensor, list[int]]:
    """Ensures batch_offsets starts with 0 and ends with the expected total length.

    This function normalizes batch offsets to ensure they follow a consistent format
    where the first element is 0 and the last element is the total length. This
    canonical format ensures that sequence i starts at batch_offsets[i] and ends
    at batch_offsets[i+1].

    Args:
        batch_offsets (Union[Tensor, list[int]]): Batch offsets that may or may not
            include 0 as the first element and/or total_length as the last element.
            Must be in ascending order.
        total_length (int): The expected total length of all concatenated sequences.
            Used to validate and potentially append to batch_offsets.

    Returns:
        Union[Tensor, list[int]]: Normalized batch offsets with 0 prepended if missing
            and total_length appended if missing. Returns the same type as input.

    Examples:
        >>> # Missing both 0 and total_length
        >>> normalize_batch_offsets(torch.tensor([5, 9]), 12)
        tensor([0, 5, 9, 12])

        >>> # Already normalized
        >>> normalize_batch_offsets(torch.tensor([0, 5, 9, 12]), 12)
        tensor([0, 5, 9, 12])
    """
    if isinstance(batch_offsets, Tensor):
        if torch.is_floating_point(batch_offsets) or torch.is_complex(batch_offsets):
            raise ValueError(
                "Expected integer tensor for batch_offsets, but got dtype "
                f"{batch_offsets.dtype}"
            )

        prepend_zero = batch_offsets[0] != 0
        append_len = batch_offsets[-1] != total_length

        if not prepend_zero and not append_len:  # Already normalized
            return batch_offsets

        if append_len and batch_offsets[-1] > total_length:
            raise ValueError(
                f"Max value of batch_offsets ({batch_offsets[-1]}) is greater than "
                f"provided total_length ({total_length})"
            )

        new_offset_len = batch_offsets.size(0) + int(prepend_zero) + int(append_len)
        out = batch_offsets.new_zeros(new_offset_len)

        copy_start = 1 if prepend_zero else 0
        copy_end = -1 if append_len else new_offset_len
        out[copy_start:copy_end] = batch_offsets
        if append_len:
            out[-1] = total_length
        return out

    else:
        prepend_zero = batch_offsets[0] != 0
        append_len = batch_offsets[-1] != total_length

        if not prepend_zero and not append_len:  # Already normalized
            return batch_offsets

        if append_len and batch_offsets[-1] > total_length:
            raise ValueError(
                f"Max value of batch_offsets ({batch_offsets[-1]}) is greater than "
                f"provided total_length ({total_length})"
            )

        out = batch_offsets.copy()
        if prepend_zero:
            out.insert(0, 0)
        if append_len:
            out.append(total_length)
        return out


@torch.jit.script
def batch_offsets_to_seq_lengths(
    batch_offsets: Union[Tensor, list[int]],
) -> Union[Tensor, list[int]]:
    """Computes sequence lengths from batch offsets.

    Given batch offsets that indicate where each sequence starts/ends in a
    concatenated tensor, this function computes the length of each individual
    sequence by taking the difference between consecutive offsets.

    Args:
        batch_offsets (Union[Tensor, list[int]]): Batch offsets tensor or list.
            Expected to be normalized (starting with 0 and ending with total length).
            Must have at least 2 elements.

    Returns:
        Union[Tensor, list[int]]: Sequence lengths for each batch element.
            Returns the same type as input. Length will be len(batch_offsets) - 1.

    Examples:
        >>> batch_offsets = torch.tensor([0, 5, 9, 12])
        >>> batch_offsets_to_seq_lengths(batch_offsets)
        tensor([5, 4, 3])

        >>> batch_offsets = [0, 3, 7, 10]
        >>> batch_offsets_to_seq_lengths(batch_offsets)
        [3, 4, 3]
    """
    if isinstance(batch_offsets, Tensor):
        return batch_offsets[1:] - batch_offsets[:-1]
    else:
        if torch.jit.is_scripting():  # type: ignore
            # Loop for Torchscript compilation
            n_seqs = len(batch_offsets) - 1
            seq_lengths: list[int] = []
            if n_seqs <= 0:
                return seq_lengths
            for i in range(n_seqs):
                seq_lengths.append(batch_offsets[i + 1] - batch_offsets[i])
            return seq_lengths
        else:
            return [
                end - start for start, end in zip(batch_offsets[:-1], batch_offsets[1:])
            ]


@torch.jit.script
def seq_lengths_to_batch_offsets(
    seq_lengths: Union[Tensor, list[int]],
) -> Union[Tensor, list[int]]:
    """Computes batch offsets from sequence lengths.

    Given the lengths of individual sequences, this function computes the batch
    offsets by cumulatively summing the lengths. The result always starts with 0
    and includes the total length as the last element.

    Args:
        seq_lengths (Union[Tensor, list[int]]): Lengths of each sequence in the batch.
            All values must be non-negative.

    Returns:
        Union[Tensor, list[int]]: Batch offsets starting with 0. Returns the same
            type as input. Length will be len(seq_lengths) + 1.

    Examples:
        >>> seq_lengths = torch.tensor([5, 4, 3])
        >>> seq_lengths_to_batch_offsets(seq_lengths)
        tensor([0, 5, 9, 12])

        >>> seq_lengths = [3, 4, 3]
        >>> seq_lengths_to_batch_offsets(seq_lengths)
        [0, 3, 7, 10]
    """
    if isinstance(seq_lengths, Tensor):
        batch_offsets_tensor = torch.zeros(
            seq_lengths.size(0) + 1, dtype=seq_lengths.dtype, device=seq_lengths.device
        )
        batch_offsets_tensor[1:] = torch.cumsum(seq_lengths, dim=0)
        return batch_offsets_tensor
    else:
        if torch.jit.is_scripting():  # type: ignore
            batch_offsets_list: list[int] = [0]
            running_sum = 0
            for length in seq_lengths:
                running_sum += length
                batch_offsets_list.append(running_sum)
            return batch_offsets_list
        else:
            return [0, *accumulate(seq_lengths, operator.add)]


@torch.jit.script
def seq_lengths_to_indices(seq_lengths: Tensor) -> Tensor:
    """Converts sequence lengths to batch indices.

    Creates a tensor where each element indicates which batch/sequence it belongs to.
    This is useful for operations that need to know the batch membership of each
    element in a concatenated tensor.

    Args:
        seq_lengths (Tensor): 1D tensor containing the length of each sequence.
            Must contain integer values. Empty tensors and 0D tensors are supported.

    Returns:
        Tensor: 1D tensor of batch indices where each position contains the index
            of the batch it belongs to. Length equals sum(seq_lengths).

    Examples:
        >>> seq_lengths = torch.tensor([5, 4])
        >>> seq_lengths_to_indices(seq_lengths)
        tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

        >>> seq_lengths = torch.tensor([2, 0, 3])  # Empty sequence in middle
        >>> seq_lengths_to_indices(seq_lengths)
        tensor([0, 0, 2, 2, 2])
    """
    assert not torch.is_floating_point(seq_lengths)
    if seq_lengths.ndim == 0:
        seq_lengths = seq_lengths.view(1)
    assert seq_lengths.ndim == 1

    n_seqs = seq_lengths.size(0)

    if n_seqs == 0:  # empty case
        return torch.empty([0], device=seq_lengths.device, dtype=seq_lengths.dtype)

    values = torch.arange(n_seqs, device=seq_lengths.device)
    out = torch.repeat_interleave(values, seq_lengths)

    return out


@torch.jit.script
def batch_offsets_to_indices(
    batch_offsets: Union[Tensor, list[int]],
    total_seq_length: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Converts batch offsets to tensor of batch indices.

    Creates a tensor where each element indicates which batch/sequence it belongs to,
    based on the provided batch offsets. This is the inverse of batch_indices_to_offsets.

    Args:
        batch_offsets (Union[Tensor, list[int]]): Batch offsets indicating where each
            sequence starts. May or may not be normalized.
        total_seq_length (Optional[int]): Total length of all sequences. If provided,
            batch_offsets will be normalized to ensure consistency.
        device (Optional[Union[torch.device, str]]): Device to place the output tensor on.
            If None, uses the device of batch_offsets if it's a Tensor, otherwise CPU.

    Returns:
        Tensor: 1D tensor of batch indices where each position contains the index
            of the batch it belongs to.

    Examples:
        >>> batch_offsets = [0, 5, 9]
        >>> batch_offsets_to_indices(batch_offsets, total_seq_length=12)
        tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

        >>> batch_offsets = torch.tensor([0, 3, 7])
        >>> batch_offsets_to_indices(batch_offsets)
        tensor([0, 0, 0, 1, 1, 1, 1])
    """
    if isinstance(device, str):
        device = torch.device(device)
    if total_seq_length is not None:
        # Normalize batch_offsets
        batch_offsets = normalize_batch_offsets(batch_offsets, total_seq_length)
    if isinstance(batch_offsets, list):
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.long, device=device)
    else:
        batch_offsets = batch_offsets.to(device=device)

    seq_lengths = batch_offsets_to_seq_lengths(batch_offsets)
    assert isinstance(seq_lengths, Tensor)
    out = seq_lengths_to_indices(seq_lengths)
    return out


@torch.jit.script
def batch_indices_to_offsets(batch_indices: Tensor) -> Tensor:
    """Converts batch indices to batch offsets.

    Given a tensor where each element indicates which batch it belongs to,
    this function computes the batch offsets by counting occurrences of each
    batch index. This is the inverse of batch_offsets_to_indices.

    Args:
        batch_indices (Tensor): 1D tensor where each element is the batch index
            that position belongs to. Must contain integer values. Batch indices
            must be contiguous and start from 0.

    Returns:
        Tensor: Batch offsets tensor starting with 0 and ending with the total
            number of elements. Length will be max(batch_indices) + 2.

    Examples:
        >>> batch_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
        >>> batch_indices_to_offsets(batch_indices)
        tensor([0, 5, 9])

        >>> # Handles empty tensors
        >>> batch_indices = torch.tensor([], dtype=torch.long)
        >>> batch_indices_to_offsets(batch_indices)
        tensor([0])
    """
    assert not torch.is_floating_point(batch_indices)

    if batch_indices.numel() == 0:  # empty case
        return torch.zeros(1, device=batch_indices.device, dtype=batch_indices.dtype)

    max_batch_index = int(batch_indices.max().item())
    batch_size = max_batch_index + 1

    counts = torch.bincount(batch_indices, minlength=batch_size)

    out = seq_lengths_to_batch_offsets(counts)
    assert isinstance(out, Tensor)

    return out


@torch.jit.script
def concatenated_to_padded(
    tensor: Tensor, batch_offsets: Tensor, pad_value: float = 0.0
) -> tuple[Tensor, Tensor]:
    """Converts concatenated sequences to batched and padded sequences.

    Takes a tensor containing concatenated sequences of varying lengths and converts
    it to a regular batched tensor with padding, along with a mask indicating
    padded positions.

    Args:
        tensor (Tensor): A tensor of shape (total_sequence_length, D1, D2, ..., Dn)
            containing concatenated sequences.
        batch_offsets (Tensor): A 1D tensor specifying where along the first dimension
            of `tensor` each sequence starts.
        pad_value (float, optional): Value to use for padding. Defaults to 0.0.

    Returns:
        out (Tensor): A tensor of shape (batch_size, max_sequence_length, D1, D2, ..., Dn)
            with sequences padded to the same length.
        padding_mask (Tensor): A boolean tensor of shape (batch_size, max_sequence_length)
            that is True at locations where `out` is padding.

    Examples:
        >>> # Convert concatenated sequences to padded format
        >>> tensor = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        >>> batch_offsets = torch.tensor([0, 2, 5])  # Two sequences: length 2 and 3
        >>> padded, mask = concatenated_to_padded(tensor, batch_offsets)
        >>> padded
        tensor([[[1, 1],
                 [2, 2],
                 [0, 0]],
                [[3, 3],
                 [4, 4],
                 [5, 5]]])
        >>> mask
        tensor([[False, False,  True],
                [False, False, False]])

        >>> # Works with higher dimensional features
        >>> tensor = torch.randn(10, 3, 4)  # 10 total timesteps, 3x4 features each
        >>> batch_offsets = torch.tensor([0, 3, 7, 10])  # 3 sequences
        >>> padded, mask = concatenated_to_padded(tensor, batch_offsets)
        >>> padded.shape
        torch.Size([3, 4, 3, 4])  # 3 batches, max length 4, 3x4 features
    """
    validate_atleast_nd(tensor, 1)
    if not batch_offsets.ndim == 1:
        raise ValueError(f"Expected batch_offsets to be 1D, got {batch_offsets.ndim}")

    # add the total length to the end of the batch offsets if needed
    batch_offsets_normed = normalize_batch_offsets(batch_offsets, tensor.shape[0])
    assert isinstance(batch_offsets_normed, Tensor)
    batch_offsets = batch_offsets_normed

    seq_lens = batch_offsets_to_seq_lengths(batch_offsets)
    assert isinstance(seq_lens, Tensor)
    batch_size = batch_offsets.shape[0] - 1
    max_len = int(torch.max(seq_lens))

    feature_dims = tensor.shape[1:]
    out_shape = (batch_size, max_len) + feature_dims

    # Fast path: If all sequences are equal length can just return a view
    if torch.all(seq_lens == max_len):
        out = tensor.reshape(out_shape)
        padding_mask = torch.zeros(
            batch_size, max_len, device=tensor.device, dtype=torch.bool
        )
        return out, padding_mask

    # Full path: Construct the padded outputs and fill them with elements from
    # the input tensor
    out = tensor.new_full(out_shape, pad_value)
    padding_mask = torch.ones(
        (batch_size, max_len), device=tensor.device, dtype=torch.bool
    )

    # Construct indices for vectorized scatter operation

    # indices pointing to the batch each token lives in
    batch_indices = batch_offsets_to_indices(batch_offsets, tensor.shape[0])

    # indices of each token's position within its batch
    arange = torch.arange(tensor.shape[0], device=tensor.device)
    token_indices = arange - batch_offsets[batch_indices]

    # Scatter into the output tensors
    out.index_put_((batch_indices, token_indices), tensor, accumulate=False)
    padding_mask.index_put_(
        (batch_indices, token_indices),
        torch.zeros_like(token_indices, dtype=torch.bool),
        accumulate=False,
    )

    return out, padding_mask


@torch.jit.script
def padded_to_concatenated(
    tensor: Tensor, padding_mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """Converts batched and padded sequences to concatenated sequences.

    Takes a regular batched tensor with padding and converts it to a concatenated
    format where all non-padded elements are concatenated along the first dimension.

    Args:
        tensor (Tensor): A tensor of shape (batch_size, max_seq_length, D1, D2, ..., Dn)
            containing batched sequences with padding.
        padding_mask (Tensor, optional): Optional boolean tensor of shape
            (batch_size, max_seq_length) where True indicates padded positions. If None,
            this function assumes that "tensor" has no padding.

    Returns:
        out (Tensor): A tensor of shape (total_seq_length, D1, D2, ..., Dn) containing
            all non-padded elements concatenated.
        batch_offsets (Tensor): A 1D tensor indicating where each batch element starts,
            including leading 0 and trailing total_seq_length.

    Examples:
        >>> # Convert padded sequences to concatenated format
        >>> padded = torch.tensor([[[1, 1], [2, 2], [0, 0]],
        ...                        [[3, 3], [4, 4], [5, 5]]])
        >>> mask = torch.tensor([[False, False, True],
        ...                      [False, False, False]])
        >>> concat, offsets = padded_to_concatenated(padded, mask)
        >>> concat
        tensor([[1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5]])
        >>> offsets
        tensor([0, 2, 5])

        >>> # Without padding mask (assumes no padding)
        >>> padded = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> concat, offsets = padded_to_concatenated(padded, None)
        >>> concat
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])
        >>> offsets
        tensor([0, 2, 4])

        >>> # Empty sequences are handled correctly
        >>> padded = torch.tensor([[[0, 0], [0, 0]], [[1, 1], [0, 0]]])
        >>> mask = torch.tensor([[True, True], [False, True]])
        >>> concat, offsets = padded_to_concatenated(padded, mask)
        >>> concat
        tensor([[1, 1]])
        >>> offsets
        tensor([0, 0, 1])
    """
    validate_atleast_nd(tensor, 2)
    batch_size, max_len = tensor.shape[:2]
    feature_dims = tensor.shape[2:]

    # Early return for empty tensor
    if batch_size == 0 or max_len == 0:
        out = torch.empty((0,) + feature_dims, dtype=tensor.dtype, device=tensor.device)
        batch_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.long, device=tensor.device
        )
        return out, batch_offsets

    # Early return for no padding: All sequences are same length so can just reshape it
    if padding_mask is None or not padding_mask.any():
        total_len = batch_size * max_len
        out_shape = (total_len,) + feature_dims
        out = tensor.reshape(out_shape)
        batch_offsets = torch.arange(0, total_len + 1, max_len, device=tensor.device)

        return out, batch_offsets

    if padding_mask.ndim != 2:
        raise ValueError(f"Expected padding_mask to be 2D, got {padding_mask.ndim}")
    if padding_mask.shape[0] != batch_size:
        raise ValueError("Batch size mismatch between tensor and padding_mask")
    if padding_mask.shape[1] != max_len:
        raise ValueError("Sequence length mismatch between tensor and padding_mask")

    nonpad_mask = padding_mask.logical_not()
    seq_lens = nonpad_mask.sum(-1).to(torch.long)
    batch_offsets = seq_lengths_to_batch_offsets(seq_lens)
    assert isinstance(batch_offsets, Tensor)
    total_len = int(batch_offsets[-1])

    out_shape = (total_len,) + feature_dims
    out = tensor.new_empty(out_shape)

    # meshgrid-like indices
    batch_indices = torch.arange(batch_size, device=tensor.device)
    batch_indices = batch_indices.unsqueeze(1).expand(batch_size, max_len)

    token_indices = torch.arange(max_len, device=tensor.device)
    token_indices = token_indices.unsqueeze(0).expand(batch_size, max_len)

    # select non-padding indices: shape sum(seq_lens)
    sel_batch_indices = batch_indices[nonpad_mask]
    sel_token_indices = token_indices[nonpad_mask]

    # Compute destination indices and extract values
    dest_indices = (batch_offsets[sel_batch_indices] + sel_token_indices).to(torch.long)
    values = tensor[nonpad_mask]

    # Scatter values into the output tensor
    out.index_put_((dest_indices,), values, accumulate=False)

    return out, batch_offsets


@torch.jit.script
def batch_dim_to_leading_index(tensor: Tensor) -> Tensor:
    """Prepends batch indices to the last dimension of a tensor.

    This function takes a batched tensor and adds the batch index as the first
    element of the last dimension for each element. This is useful for operations
    that need to track which batch each element came from after flattening.

    In particular, this function is useful for preparing the indices tensor used
    in construction of a torch.sparse_coo_tensor.

    Args:
        tensor (Tensor): Input tensor of shape (batch_size, D1, D2, ..., Dn).
            Must have at least 2 dimensions.

    Returns:
        Tensor: Output tensor of shape (batch_size, D1, D2, ..., Dn+1) where
            the first element of the last dimension is the batch index.

    Examples:
        >>> # 2D tensor
        >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        >>> batch_dim_to_leading_index(x)
        tensor([[[0, 1, 2],
                 [0, 3, 4]],
                [[1, 1, 2],
                 [1, 3, 4]],
                [[2, 5, 6],
                 [2, 5, 6]]])

        >>> # 3D tensor
        >>> x = torch.randn(2, 3, 4)
        >>> result = batch_dim_to_leading_index(x)
        >>> result.shape
        torch.Size([2, 3, 5])
    """
    batch_size = tensor.shape[0]
    last_dim = tensor.shape[-1]
    other_dims = torch._shape_as_tensor(tensor)[1:-1].to(tensor.device)
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=tensor.device), torch.prod(other_dims), 0
    )
    flattened = torch.concat([batch_index.unsqueeze(-1), tensor.view(-1, last_dim)], -1)
    new_shape = list(tensor.shape)
    new_shape[-1] = last_dim + 1
    return flattened.reshape(new_shape)


@torch.jit.script
def batch_offsets_from_sparse_tensor_indices(indices_tensor: Tensor) -> Tensor:
    """Gets the batch offsets from an index tensor where the first element of the
    first dimension is the batch index.

    This function is typically used with the indices() tensor of a sparse COO tensor
    to extract batch offset information, assuming the first index dimension represents
    the batch dimension.

    Args:
        indices_tensor (Tensor): A tensor of shape (M x nnz), where M is
            the number of dimensions of the underlying sparse tensor and nnz is the
            number of nonzero elements in the sparse tensor. Assumes the sparse
            tensor has been coalesce()d.

    Returns:
        Tensor: A 1D tensor with elements corresponding to the first
            incidence of each unique element in the first position of the M axis,
            i.e., the batch offsets if the first element is the batch index.
            Includes leading 0 and trailing nnz.

    Examples:
        >>> # Extract batch offsets from sparse tensor indices
        >>> indices = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2],
        ...                         [0, 1, 2, 0, 2, 1, 2, 3]])
        >>> offsets = batch_offsets_from_sparse_tensor_indices(indices)
        >>> offsets
        tensor([0, 3, 5, 8])

        >>> # Create a sparse tensor and extract offsets from its indices
        >>> indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 2]])
        >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 3))
        >>> offsets = batch_offsets_from_sparse_tensor_indices(sparse.indices())
        >>> offsets
        tensor([0, 2, 4])
    """
    assert not torch.is_floating_point(indices_tensor)

    if indices_tensor.numel() == 0:  # empty case
        return torch.zeros(1, device=indices_tensor.device, dtype=indices_tensor.dtype)

    batch_indices = indices_tensor[0]
    out = batch_indices_to_offsets(batch_indices)

    return out


@torch.jit.script
def sparse_tensor_to_padded(tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Converts a sparse COO tensor to padded dense format.

    This function takes a sparse tensor in COO format and converts it to a padded
    dense representation. The sparse tensor is assumed to have its first index
    dimension representing the batch dimension.

    Args:
        tensor (Tensor): Sparse COO tensor where the first index dimension is the
            batch dimension. Must be coalesced.

    Returns:
        batched_tensor (Tensor): Dense values in padded format with shape
            (batch_size, max_seq_length, value_dims...)
        batched_indices (Tensor): Indices in padded format with shape
            (batch_size, max_seq_length, num_index_dims)
        pad_mask (Tensor): Boolean mask indicating padded positions with shape
            (batch_size, max_seq_length)

    Examples:
        >>> indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 2]])
        >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 3))
        >>> dense_vals, dense_idx, mask = sparse_tensor_to_padded(sparse)
        >>> dense_vals.shape
        torch.Size([2, 2])
    """
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")
    batch_offsets = batch_offsets_from_sparse_tensor_indices(tensor.indices())
    batched_tensor, pad_mask = concatenated_to_padded(tensor.values(), batch_offsets)
    batched_indices, pad_mask_2 = concatenated_to_padded(
        tensor.indices().T, batch_offsets
    )
    assert torch.equal(pad_mask, pad_mask_2)
    return batched_tensor, batched_indices, pad_mask


@torch.jit.script
def padded_to_sparse_tensor(
    batched_values: Tensor,
    batched_indices: Tensor,
    pad_mask: Tensor,
    sparse_tensor_shape: Optional[list[int]] = None,
) -> Tensor:
    """Converts padded dense format back to a sparse COO tensor.

    This function takes values and indices in padded dense format and converts
    them back to a sparse COO tensor, removing any padding.

    Args:
        batched_values (Tensor): Values in padded format with shape
            (batch_size, max_seq_length, value_dims...)
        batched_indices (Tensor): Indices in padded format with shape
            (batch_size, max_seq_length, num_index_dims)
        pad_mask (Tensor): Boolean mask indicating padded positions with shape
            (batch_size, max_seq_length). True indicates padding.
        sparse_tensor_shape (Optional(list[int])): Shape of the output sparse tensor.
            If None, the resulting sparse tensor shape will be inferred by the
            torch.sparse_coo_tensor constructor.

    Returns:
        Tensor: Sparse COO tensor with the specified shape, with padding removed.

    Examples:
        >>> values = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        >>> indices = torch.tensor([[[0, 0], [0, 1]], [[1, 0], [0, 0]]])
        >>> mask = torch.tensor([[False, False], [False, True]])
        >>> sparse = padded_to_sparse_tensor(values, indices, mask, [2, 3])
        >>> sparse.to_dense()
        tensor([[1., 2., 0.],
                [3., 0., 0.]])
    """
    stacked_values, batch_offsets = padded_to_concatenated(batched_values, pad_mask)
    stacked_indices, batch_offsets_2 = padded_to_concatenated(batched_indices, pad_mask)
    assert torch.equal(batch_offsets, batch_offsets_2)
    if sparse_tensor_shape is not None:
        return torch.sparse_coo_tensor(
            stacked_indices.T, stacked_values, sparse_tensor_shape
        ).coalesce()
    else:
        return torch.sparse_coo_tensor(stacked_indices.T, stacked_values).coalesce()


@torch.jit.script
def sparse_tensor_to_concatenated(tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Converts a sparse COO tensor to concatenated format.

    This function extracts the values and indices from a sparse tensor and returns
    them in concatenated format along with batch offsets. The sparse tensor is
    assumed to have its first index dimension representing the batch dimension.

    Args:
        tensor (Tensor): Sparse COO tensor where the first index dimension is the
            batch dimension. Should be coalesced for correct results.

    Returns:
        - values (Tensor): Concatenated values with shape (total_nnz, value_dims...)
        - indices (Tensor): Concatenated indices with shape (total_nnz, num_index_dims)
        - batch_offsets (Tensor): 1D tensor indicating where each batch starts/ends

    Examples:
        >>> indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 2]])
        >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> sparse = torch.sparse_coo_tensor(indices, values, (2, 3))
        >>> vals, idx, offsets = sparse_tensor_to_concatenated(sparse)
        >>> vals
        tensor([1., 2., 3., 4.])
        >>> offsets
        tensor([0, 2, 4])
    """
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")
    batch_offsets = batch_offsets_from_sparse_tensor_indices(tensor.indices())
    return tensor.values(), tensor.indices().T, batch_offsets


@torch.jit.script
def concatenated_to_sparse_tensor(
    values: Tensor, indices: Tensor, sparse_tensor_shape: Optional[list[int]] = None
) -> Tensor:
    """Creates a sparse COO tensor from concatenated values and indices.

    This is a simple wrapper around torch.sparse_coo_tensor that transposes
    the indices to the expected format.

    Args:
        values (Tensor): Concatenated values with shape (total_nnz, value_dims...)
        indices (Tensor): Concatenated indices with shape (total_nnz, num_index_dims).
            Each row contains the indices for one non-zero element.
        sparse_tensor_shape (Optional[list[int]]): Shape of the output sparse tensor.
            If None, the shape is inferred from the maximum indices.

    Returns:
        Tensor: Sparse COO tensor constructed from the provided values and indices.

    Examples:
        >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> indices = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 2]])
        >>> sparse = concatenated_to_sparse_tensor(values, indices, [2, 3])
        >>> sparse.to_dense()
        tensor([[1., 2., 0.],
                [3., 0., 4.]])
    """
    if sparse_tensor_shape is not None:
        return torch.sparse_coo_tensor(
            indices.T, values, sparse_tensor_shape
        ).coalesce()
    else:
        return torch.sparse_coo_tensor(indices.T, values).coalesce()
