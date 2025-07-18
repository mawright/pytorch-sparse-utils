import math

import random
import hypothesis
import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings, example
from hypothesis import strategies as st
from torch import Tensor, nn

from pytorch_sparse_utils.shape_ops import sparse_reshape
from pytorch_sparse_utils.indexing.script_funcs import flatten_nd_indices

from .. import random_sparse_tensor, random_sparse_tensor_strategy


def sparse_1d_tensor(device: str):
    """Sparse tensor [0.0, 4.0, 0.0, 3.0]"""
    return torch.sparse_coo_tensor(
        torch.tensor([[1, 3]], device=device),
        torch.tensor([4.0, 3.0], device=device),
        size=(4,),
    )


def sparse_1d1d_tensor(device: str):
    """Sparse tensor with 1 sparse dim and 1 dense dim."""
    return torch.sparse_coo_tensor(
        torch.tensor([[1]], device=device),
        torch.arange(4.0, dtype=torch.float, device=device).unsqueeze(0),
        size=(2, 4),
    )


@pytest.mark.cpu_and_cuda
class TestSparseReshape:
    def test_unit_reshape_sparse(self, device: str):
        """Test reshaping a tensor from [2, 3] to [3, 2]"""
        indices = torch.tensor([[0, 1], [0, 2], [1, 0]], device=device).T
        values = torch.tensor([10, 20, 30], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        out = sparse_reshape(tensor, new_sparse_shape=[3, 2])

        expected_indices = torch.tensor([[0, 1], [1, 0], [1, 1]], device=device).T

        assert out.shape == (3, 2)
        assert torch.equal(out.indices(), expected_indices)
        assert torch.equal(out.values(), values)

    def test_unit_reshape_dense(self, device: str):
        """Test reshaping a hybrid sparse tensor's dense dim from 1D to 2D"""
        indices = torch.tensor([[1]], device=device)
        values = torch.tensor([[0, 1, 2, 3]], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 4))

        out = sparse_reshape(tensor, new_dense_shape=[2, 2])

        assert out.shape == (2, 2, 2)
        assert out.dense_dim() == 2
        assert torch.equal(out.indices(), indices)
        assert torch.equal(out.values(), values.reshape(1, 2, 2))

    def test_error_not_sparse(self, device: str):
        """Test erroring when passed a dense tensor."""
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="Received non-sparse tensor",
        ):
            sparse_reshape(torch.randn(3, 4, device=device), new_sparse_shape=[4, 3])

    def test_error_no_new_shape(self, device: str):
        """Test error with no new shape"""
        tensor = sparse_1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="Expected one or both",
        ):
            sparse_reshape(tensor)

    @pytest.mark.parametrize(
        "kind, kwarg",
        [
            ("sparse", {"new_sparse_shape": [2, -2]}),
            ("dense", {"new_dense_shape": [3, -5]}),
        ],
    )
    def test_error_invalid_dim_value(
        self, kind: str, kwarg: dict[str, list[int]], device: str
    ):
        """Test invalid dims"""
        tensor = sparse_1d1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match=f"Invalid {kind} shape dimension",
        ):
            sparse_reshape(tensor, **kwarg)

    @pytest.mark.parametrize(
        "kind, kwarg",
        [
            ("sparse", {"new_sparse_shape": [-1, -1]}),
            ("dense", {"new_dense_shape": [-1, -1]}),
        ],
    )
    def test_error_gt_1_inferred(
        self, kind: str, kwarg: dict[str, list[int]], device: str
    ):
        """Test trying to infer more than one dim"""
        tensor = sparse_1d1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="Only one dimension can be inferred",
        ):
            sparse_reshape(tensor, **kwarg)

    def test_not_divisible_size_sparse(self, device):
        tensor = sparse_1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, RuntimeError),  # pyright: ignore[reportArgumentType]
            match="invalid for input with sparse shape",
        ):
            sparse_reshape(tensor, new_sparse_shape=[-1, 3])

    def test_not_divisible_size_dense(self, device):
        tensor = sparse_1d1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, RuntimeError),  # pyright: ignore[reportArgumentType]
            match="invalid for input with dense shape",
        ):
            sparse_reshape(tensor, new_dense_shape=[-1, 3])

    def test_wrong_numel_sparse(self, device):
        tensor = sparse_1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, RuntimeError),  # pyright: ignore[reportArgumentType]
            match="invalid for input with sparse shape",
        ):
            sparse_reshape(tensor, new_sparse_shape=[100])

    def test_wrong_numel_dense(self, device):
        tensor = sparse_1d1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, RuntimeError),  # pyright: ignore[reportArgumentType]
            match="invalid for input with dense shape",
        ):
            sparse_reshape(tensor, new_dense_shape=[100])

    @example(
        tensor_config={
            "sparse_shape": [0],
            "dense_shape": [],
            "sparsity": 0.0,
            "seed": 0,
            "dtype": torch.float32,
        },
        reshape_sparse=True,  # or any other generated value
        reshape_dense=True,  # or any other generated value
        infer_sparse=False,
        infer_dense=False,  # or any other generated value
        new_shape_seed=0,
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    @given(
        tensor_config=random_sparse_tensor_strategy(max_dim=10),
        reshape_sparse=st.booleans(),
        reshape_dense=st.booleans(),
        infer_sparse=st.booleans(),
        infer_dense=st.booleans(),
        new_shape_seed=st.integers(0, int(1e10)),
    )
    def test_hypothesis(
        self,
        tensor_config: dict,
        reshape_sparse: bool,
        reshape_dense: bool,
        infer_sparse: bool,
        infer_dense: bool,
        new_shape_seed: int,
        device: str,
    ):
        """Property-based test with hypothesis.

        1. Draws a random sparse tensor
        2. Generates new compatible sparse and/or dense shapes
        3. Compares against reshaped dense tensor
        """
        assume(reshape_sparse or reshape_dense)
        tensor = random_sparse_tensor(**tensor_config, device=device)

        sparse_shape_in = list(tensor.shape[: tensor.sparse_dim()])
        dense_shape_in = list(tensor.shape[tensor.sparse_dim() :])

        new_sparse_shape, new_dense_shape = None, None

        if reshape_sparse:
            new_sparse_shape = random_new_shape(
                sparse_shape_in, new_shape_seed, infer_one_dim=infer_sparse
            )
        if reshape_dense:
            new_dense_shape = random_new_shape(
                dense_shape_in, new_shape_seed, infer_one_dim=infer_dense
            )

        out = sparse_reshape(tensor, new_sparse_shape, new_dense_shape)

        # Property tests
        # Same nnz
        assert out._nnz() == tensor._nnz()

        # Values only reshaped
        assert torch.equal(out.values().reshape_as(tensor.values()), tensor.values())

        # Compute expected output shape
        if new_sparse_shape is not None:
            if infer_sparse:
                sparse_numel = math.prod(sparse_shape_in)
                if sparse_numel == 0:
                    inferred_dim = 0
                else:
                    partial_inferred = math.prod([d for d in new_sparse_shape if d != -1])
                    inferred_dim = sparse_numel // partial_inferred
                new_sparse_shape[new_sparse_shape.index(-1)] = inferred_dim
            expected_shape = new_sparse_shape.copy()
        else:
            expected_shape = sparse_shape_in.copy()
        if new_dense_shape is not None:
            if infer_dense:
                dense_numel = math.prod(dense_shape_in)
                if dense_numel == 0:
                    inferred_dim = 0
                else:
                    partial_inferred = math.prod([d for d in new_dense_shape if d != -1])
                    inferred_dim = dense_numel // partial_inferred
                new_dense_shape[new_dense_shape.index(-1)] = inferred_dim
            expected_shape += new_dense_shape
        else:
            expected_shape += dense_shape_in

        # Flat indices the same
        if reshape_sparse:
            in_flat_indices = flatten_nd_indices(
                tensor.indices(), tensor.new_tensor(sparse_shape_in)
            )[0]
            new_flat_indices = flatten_nd_indices(
                out.indices(), out.new_tensor(new_sparse_shape)
            )[0]
            assert torch.equal(in_flat_indices, new_flat_indices)
        else:
            assert torch.equal(tensor.indices(), out.indices())

        # Final shape the same
        assert list(out.shape) == expected_shape

        # Compare against dense tensor
        tensor_dense = tensor.to_dense()
        out_dense = tensor_dense.reshape(out.shape)
        assert torch.equal(out_dense, out.to_dense())


def _divisors(n: int) -> list[int]:
    """Get all divisors of n"""
    if n == 0:
        return [0]

    root = math.isqrt(n)
    small = [d for d in range(1, root + 1) if n % d == 0]
    large = [n // d for d in reversed(small) if d * d != n]
    return small + large


def random_new_shape(
    in_shape: list[int], seed: int, max_ndim: int = 6, infer_one_dim: bool = False
) -> list[int]:
    numel = math.prod(in_shape)
    rng = random.Random(seed)

    target_ndim = rng.randint(1, max_ndim)

    if numel == 0:
        dims = [rng.randint(1, 10) for _ in range(target_ndim)]
        n_zeros = rng.randint(1, len(dims))
        for idx in rng.sample(range(len(dims)), n_zeros):
            dims[idx] = 0
    else:
        dims: list[int] = []
        remaining = numel
        for _ in range(target_ndim):
            d = rng.choice(_divisors(remaining))
            dims.append(d)
            if d > 0:
                remaining //= d
            else:
                assert remaining == 0

        dims.append(remaining)
        rng.shuffle(dims)

    if infer_one_dim:
        m1_index = rng.randint(0, len(dims) - 1)
        dims[m1_index] = -1

    return dims
