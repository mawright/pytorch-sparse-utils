# pytorch-sparse-utils documentation

Welcome to the documentation for pytorch-sparse-utils.

For basic information, please see the [repository Readme](https://github.com/mawright/pytorch-sparse-utils).

## Motivation
While PyTorch tensors have native support for sparse formats, many advanced indexing, updating, and processing patterns needed for modern neural-net architectures are not fully supported, especially in the context of an autograd graph for training. To cherry-pick an example, as of the latest PyTorch version (2.7.1), simple slice selection is not supported for sparse tensors, even in forward mode:
```python
import torch
X = torch.sparse_coo_tensor(
    torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]]).T,
    torch.randn(4),
    size=(5, 5),
    requires_grad=True
)
data = X[0, :]
```

Output:
```
NotImplementedError: Could not run 'aten::as_strided' with arguments from the 'SparseCPU' backend.
...
```

The `pytorch-sparse-utils` package features low-level utilities developed at [Berkeley Lab](https://www.lbl.gov) as part of a project using PyTorch sparse tensors for large, sparse, irregularly-structured scientific images, with an eye towards modern neural networks like Transformers. The principles of the project include:

- **Performance:** Since indexing, scattering, and similar operations can be called dozens of times per training iteration, performance of these basic operations are critical. `pytorch-sparse-utils` has universal TorchScript coverage, providing the fastest feasible pure-PyTorch implementations, with a clear roadmap for further optimization through future ports to C++/CUDA extensions.
- **Flexibility:** Sparse data exist not just in 2D images or 3D volumes, but in arbitrary-dimensional formats across diverse scientific and ML applications. `pytorch-sparse-utils` provides dimension-agnostic operations that work with 1D sequences, 2D matrices, 3D volumes, and higher-dimensional tensors and arrays, providing a universal interface for researchers across domains.
- **Correctness:** Low-level utilities form the building blocks for basic data handling operations and complex ML workflows, making their correctness absolutely critical and any bugs potentially difficult to isolate. `pytorch-sparse-utils` ensures correctness through comprehensive unit tests and property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/en/latest/), ensuring operations handle standard inputs as well as expected and unexpected edge cases.

## 