# Indexing utilties

The indexing module provides operations for selecting, slicing, and gather/scattering sparse tensors.

## Basic Operations
::: indexing
    options:
        members:
            - sparse_select
            - sparse_index_select
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## Bulk Indexing

::: indexing
    options:
        members:
            - batch_sparse_index
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## Sparse Tensor Scatter

::: indexing
    options:
        members:
            - scatter_to_sparse_tensor
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## Miscellaneous Functions

::: indexing
    options:
        members:
            - unique_rows
            - union_sparse_indices
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## Indexing Helpers

::: indexing
    options:
        members:
            - flatten_nd_indices
            - unflatten_nd_indices
            - flatten_sparse_indices
            - linearize_sparse_and_index_tensors
            - get_sparse_index_mapping
            - gather_mask_and_fill
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3