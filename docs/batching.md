# Batching utilities

When dealing with sparse data, examples (images, volumes, etc.) often have different amounts of nonzero elements per instance. While most sparse formats can handle heterogenous-sized batches naturally, most neural network contexts expect dense tensors with regularly-strided batches.

Most often, differently-sized batches are handled via padding, with junk data concatenated onto shorter examples so a batch of examples can be stacked into a regularly-sized tensor. Alternatively, the examples can simply be concatenated together, with accompanying metadata specifying the start index of each example along the concatenated batch dimension. This is the internal format for implementations like PyTorch's NestedTensors and TensorFlow's RaggedTensors. 

When data are large, the concatenated-batch format can save significant amounts of computation and memory compared to padding while being mathematically equivalent for batch-invariant operations like FFNs and LayerNorm. For batch-dependent operations like Multi-head Attention, we take the approach of switching to a padded format immediately before calling the Multi-head Attention kernel that expects a padded format, and converting back to concatenated-batch format immediately after.

In `pytorch-sparse-utils`'s batching module, we have utilities for converting between sparse tensors, concatenated-batch format, and traditional padded format.

Like the rest of `pytorch-sparse-utils`, all batching conversion utilities are TorchScript-accelerated and optimized to minimize computation time and memory use.

## Concatenated-batch format
The concatenated-batch format mentioned above consists of a tensor where all examples are concatenated along the first (i.e., sequence) dimension, accompanied by a `batch_offsets` tensor that tracks where each example begins and ends.

### Format specification
- **Concatenated tensor**: Shape `(total_length, D1, D2, ..., Dn)`, where `total_length` is the sum of all individual sequence lengths.
- **Batch offsets**: 1D tensor containing the starting/ending indices of each batch in the concatenated tensor. Can be provided in multiple formats:
    - With or without leading 0
    - With or without trailing `total_length`.

Canonically, the batch offsets tensor includes both, ensuring that sequence `i` in the concatenated tensor begins at index `batch_offsets[i]` and ends at `batch_offsets[i+1]`. The function `normalize_batch_offsets` is used to add the leading 0 and trailing `total_length` if not present.

## Format conversions
::: batching
    options:
        members:
            - concatenated_to_padded
            - padded_to_concatenated
            - sparse_tensor_to_concatenated
            - concatenated_to_sparse_tensor
            - sparse_tensor_to_padded
            - padded_to_sparse_tensor
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## Offset and index conversions
::: batching
    options:
        members:
            - normalize_batch_offsets
            - batch_offsets_to_seq_lengths
            - seq_lengths_to_batch_offsets
            - batch_offsets_to_indices
            - seq_lengths_to_indices
            - batch_indices_to_offsets
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## Other utilties
::: batching
    options:
        members:
            - split_batch_concatenated_tensor
            - batch_dim_to_leading_index
            - batch_offsets_from_sparse_tensor_indices
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3