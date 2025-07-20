# Sparse tensor shape operations

The `shape_ops` module contains sparse tensor versions of common reshaping operations like `reshape` and `flatten`, which are not implemented for sparse tensors in base PyTorch.

::: shape_ops
    options:
        members:
            - sparse_reshape
            - sparse_flatten
            - sparse_squeeze
            - sparse_resize
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        heading_level: 3