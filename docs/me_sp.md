# Third-party sparse library integration

`pytorch-sparse-utils` features integrations with three major libraries for sparse arrays and tensors:

- [Pydata sparse](https://sparse.pydata.org/), a numpy-like sparse array implementation with close numpy integration.
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), an Nvidia library for convolutions on sparse tensors.
- [spconv](https://github.com/traveller59/spconv), another library for convolutions on sparse tensors.

All three libraries feature their own sparse tensor object formats that are distinct from the built-in PyTorch sparse tensors. `pytorch_sparse_utils`'s `conversion` module provides simple utilities to convert between the formats.

Using these conversion utilities allows for, for example, a pipeline where images are loaded as Pydata Sparse COO arrays, converted to PyTorch sparse tensors in a `torch.utils.DataLoader`, converted to `MinkowskiEngine` `SparseTensors` for processing through a `MinkowskiEngine` CNN backbone, then converted back to PyTorch sparse tensors for processing with a Transformer module.

---

## Pydata sparse conversions
::: conversion
    options:
        members:
            - torch_sparse_to_pydata_sparse
            - pydata_sparse_to_torch_sparse
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## MinkowskiEngine conversions
::: conversion
    options:
        members:
            - torch_sparse_to_minkowski
            - minkowski_to_torch_sparse
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3

---

## spconv conversions
::: conversion
    options:
        members:
            - torch_sparse_to_spconv
            - spconv_to_torch_sparse
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3