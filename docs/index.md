# pytorch-sparse-utils documentation

Welcome to the documentation for pytorch-sparse-utils.

For basic information, please see the [repository Readme](https://github.com/mawright/pytorch-sparse-utils).

## Introduction
While PyTorch tensors have native support for sparse formats, many advanced indexing, updating, and processing patterns needed for modern neural-net architectures are not fully supported, especially in the context of an autograd graph for training. To cherry-pick an example, as of the latest PyTorch version (2.7.1), reshaping is not supported for sparse tensors:
```python
import torch
X = torch.sparse_coo_tensor(
    torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]]).T,
    torch.randn(4),
    size=(5, 5),
    requires_grad=True
)
X_reshaped = X.reshape(-1)
```

Output:
```txt
RuntimeError: reshape is not implemented for sparse tensors
```

The `pytorch-sparse-utils` package features low-level utilities developed at [Berkeley Lab](https://www.lbl.gov) as part of a project using PyTorch sparse tensors for large, sparse, irregularly-structured scientific images, with an eye towards modern neural networks like Transformers. The principles of the project include:

- **Performance:** Since indexing, scattering, and similar operations can be called dozens of times per training iteration, performance of these basic operations are critical. `pytorch-sparse-utils` has universal TorchScript coverage, providing the fastest feasible pure-PyTorch implementations, with a clear roadmap for further optimization through future ports to C++/CUDA extensions.
- **Flexibility:** Sparse data exist not just in 2D images or 3D volumes, but in arbitrary-dimensional formats across diverse scientific and ML applications. `pytorch-sparse-utils` provides dimension-agnostic operations that work with 1D sequences, 2D matrices, 3D volumes, and higher-dimensional tensors and arrays, providing a universal interface for researchers across domains.
- **Familiarity:** Sparse data structures like COO tensors often require different usage patterns than standard dense tensors due to their differeng implementation. `pytorch-sparse-utils` attempts to make sparse tensors as easy to use as dense tensors by abstracting away these semantic differences into equivalent APIs like `sparse_reshape` and `batch_sparse_index`.
- **Correctness:** Low-level utilities form the building blocks for basic data handling operations and complex ML workflows, making their correctness absolutely critical and any bugs potentially difficult to isolate. `pytorch-sparse-utils` ensures correctness through comprehensive unit tests and property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/en/latest/), ensuring operations handle standard inputs as well as expected and unexpected edge cases.

## Feature Overview
`pytorch-sparse-utils` contains various sparse-tensor-specific utilities meant to bring use and manipulation of sparse tensors closer to feature parity with dense tensors. Example functions include:

- `sparse_reshape`, to conveniently reshape the sparse and/or dense dimensions of a (hybrid) sparse tensor, along with the special-case functions `sparse_squeeze` and `sparse_flatten`.
- `sparse_index_select`, with identical API to the built-in `.index_select` but with enhanced autograd support.
- `batch_sparse_index`, a performant gather-type operation for bulk selection from a sparse tensor, particularly useful for randomly accessing embeddings within a sparse tensor.

For more information, see the rest of this documentation.

## Contributions
The initial release of `pytorch-sparse-utils` represents the work of one author, but contributions are welcome. Feel free to open pull requests, give feedback, report bugs, or request additional features on the project's [GitHub repository](https://github.com/mawright/pytorch-sparse-utils).