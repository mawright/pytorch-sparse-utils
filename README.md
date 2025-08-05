# pytorch-sparse-utils

[![Tests](https://github.com/mawright/pytorch-sparse-utils/actions/workflows/tests.yml/badge.svg)](https://github.com/mawright/pytorch-sparse-utils/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mawright/pytorch-sparse-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/mawright/pytorch-sparse-utils)
[![Documentation Status](https://github.com/mawright/pytorch-sparse-utils/actions/workflows/docs.yml/badge.svg)](https://mawright.github.io/pytorch-sparse-utils/)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![License](https://img.shields.io/github/license/mawright/pytorch-sparse-utils)](https://github.com/mawright/pytorch-sparse-utils/blob/main/LICENSE)

Low-level utilities for PyTorch sparse tensors and operations.

## Introduction

PyTorch's implementation of sparse tensors is lacking full support for many common operations. This repository contains a set of utilities for making PyTorch sparse tensors into more usable general-purpose sparse data structures, particularly in the context of modern neural network architectures like Transformer-based models.

For example, while the basic operation `index_select` has a sparse forward implementation, using it as part of an autograd graph alongside direct manipulation of the sparse tensor's values is not supported:

```python
# Latest PyTorch version (2.7.1) as of this writing
X = torch.sparse_coo_tensor(
    torch.randint(0, 11, size=(3, 10)),
    torch.randn(10, 2),
    size=(10, 10, 10, 2),
    requires_grad=True
)
selected = X.index_select(0, torch.tensor([0, 1]))  # Sparse tensor
data = selected.values()  # Dense tensor
loss = data.sum()
loss.backward()
print(X.grad)
```

Partial output (error message continues):
```
Traceback (most recent call last):
  File "/global/u2/m/mwright/Code/pytorch-sparse-utils/demo.py", line 13, in <module>
    loss.backward()
  File "/global/homes/m/mwright/.conda/envs/me_121/lib/python3.11/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "/global/homes/m/mwright/.conda/envs/me_121/lib/python3.11/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/global/homes/m/mwright/.conda/envs/me_121/lib/python3.11/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NotImplementedError: Could not run 'aten::index_add_' with arguments from the 'SparseCPU' backend...
...
```

One utility provided by pytorch-sparse-utils is the function `sparse_index_select`, which functions identically to vanilla `index_select` on sparse tensors but integrates seamlessly with autograd:

```python
from pytorch_sparse_utils.indexing import sparse_index_select

selected = sparse_index_select(X, 0, torch.tensor([0, 1]))
data = selected.values()
loss = data.sum()
loss.backward()
print(X.grad)
```

Output:

```
tensor(indices=tensor([[0, 1, 2, 3],
                       [0, 1, 2, 3]]),
       values=tensor([1., 1., 0., 0.]),
       size=(5, 5), nnz=4, layout=torch.sparse_coo)
```

This repository differs from similar packages like [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) in that we build on top of the built-in PyTorch sparse tensor classes.
This allows compatibility with the supported built-in PyTorch operations as well as maintaining the arbitrary dimensionality features of the built-in sparse tensors, which can be essential for complex-structured data objects like the multi-level feature pyramids used for object detection and other tasks in sparse images.
It also leaves open the possibility of eventual integration into upstream PyTorch to ensure the broadest availability.

## Feature Overview

- Autograd-compatible implementations of bulk indexing, sparse tensor shape manipulations, and quick conversions between sparse tensor format and concatenated-batch format for use with position-invariant layers (Linear, BatchNorm, etc.).
- Interoperability with [Pydata sparse](https://sparse.pydata.org/), a numpy-like sparse array implementation, as well as [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [spconv](https://github.com/traveller59/spconv), two popular PyTorch libraries for convolutions on sparse images and volumes.
- Full TorchScript compatibility for performance.
- Extensive unit and property-based tests to ensure correctness and reliability.

## Installation

pytorch-sparse-utils has minimal requirements beyond PyTorch itself. The simplest way to install is to clone this repository and use `pip install`:

```bash
git clone https://github.com/mawright/pytorch-sparse-utils
cd pytorch-sparse-utils
pip install -e .  # editable installation
```

To run the test suite, you'll need to install the optional dependencies:

```bash
pip install -e ".[tests]"
```

Due to incompatibilities with newer CUDA versions, MinkowskiEngine and spconv are not installed as part of the base install. For more information on installing those libraries, see their own repositories.

## Documentation

Full documentation is available on [GitHub Pages](https://mawright.github.io/pytorch-sparse-utils/).

## See Also

pytorch-sparse-utils represents a base set of tools for more complex neural-net operations on sparse tensors. For more sparse tensor applications, see the following repositories:

- [nd-rotary-encodings](https://github.com/mawright/nd-rotary-encodings): Fast and memory-efficient rotary positional encodings (RoPE) in PyTorch, with novel algorithm updates for multi-level feature pyramids for object detection and other applications.
- [sparse-transformer-layers](https://github.com/mawright/sparse-transformer-layers): Implementations of Transformer layers tailored to sparse tensors, including variants like Multi-scale Deformable Attention. Features custom gradient checkpointing logic to effectively handle sparse tensors with potentially many nonzero entries.

## Future Plans

- Custom C++/CUDA extensions for the most performance-critical operations
- Performance benchmarks
- Expanded documentation
- Additional functionality (feature requests welcome)
- (Potentially) upstream contributions to base PyTorch