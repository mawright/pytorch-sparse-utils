import math
import random
from typing import Optional

import pytest
import torch
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from torch import Tensor

from pytorch_sparse_utils.indexing.utils import flatten_nd_indices
from pytorch_sparse_utils.shape_ops import (
    sparse_flatten,
    sparse_reshape,
    sparse_resize,
    sparse_squeeze,
)

from . import random_sparse_tensor, random_sparse_tensor_strategy


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
                    partial_inferred = math.prod(
                        [d for d in new_sparse_shape if d != -1]
                    )
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
                    partial_inferred = math.prod(
                        [d for d in new_dense_shape if d != -1]
                    )
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


@pytest.mark.cpu_and_cuda
class TestSparseFlatten:
    def test_unit_flatten_2d_to_1d(self, device: str):
        """Test flattening a 2D sparse tensor to 1D"""
        indices = torch.tensor([[0, 1, 1], [1, 0, 1]], device=device)
        values = torch.tensor([1.0, 2.0, 3.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2))

        out = sparse_flatten(tensor, start_axis=0, end_axis=1)

        # Expected: positions (0,1), (1,0), (1,1) -> 1, 2, 3
        expected_indices = torch.tensor([[1, 2, 3]], device=device)

        assert out.shape == (4,)
        assert torch.equal(out.indices(), expected_indices)
        assert torch.equal(out.values(), values)

    def test_unit_flatten_3d_middle_dims(self, device: str):
        """Test flattening middle dimensions of a 3D tensor"""
        indices = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1]], device=device)
        values = torch.tensor([1.0, 2.0, 3.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2, 2))

        out = sparse_flatten(tensor, start_axis=1, end_axis=2)

        # Shape should be (2, 4)
        assert out.shape == (2, 4)
        assert torch.equal(out.to_dense(), tensor.to_dense().reshape(2, 4))

    def test_unit_flatten_negative_indexing(self, device: str):
        """Test flattening with negative indices"""
        indices = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1]], device=device)
        values = torch.tensor([1.0, 2.0, 3.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2, 2))

        out = sparse_flatten(tensor, start_axis=-2, end_axis=-1)

        assert out.shape == (2, 4)
        assert torch.equal(out.to_dense(), tensor.to_dense().reshape(2, 4))

    def test_unit_flatten_all_dims(self, device: str):
        """Test flattening all dimensions"""
        indices = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1]], device=device)
        values = torch.tensor([1.0, 2.0, 3.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2, 2))

        out = sparse_flatten(tensor, start_axis=0, end_axis=-1)

        assert out.shape == (8,)
        assert torch.equal(out.to_dense(), tensor.to_dense().reshape(8))

    def test_error_not_sparse(self, device: str):
        """Test error when passed a dense tensor"""
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="Received non-sparse tensor",
        ):
            sparse_flatten(torch.randn(3, 4, device=device), 0, 1)

    def test_error_invalid_start_axis(self, device: str):
        """Test error with invalid start axis"""
        tensor = sparse_1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, IndexError),  # pyright: ignore[reportArgumentType]
            match="Dimension out of range",
        ):
            sparse_flatten(tensor, start_axis=5, end_axis=0)

    def test_error_invalid_end_axis(self, device: str):
        """Test error with invalid end axis"""
        tensor = sparse_1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, IndexError),  # pyright: ignore[reportArgumentType]
            match="Dimension out of range",
        ):
            sparse_flatten(tensor, start_axis=0, end_axis=5)

    def test_error_end_not_greater_than_start(self, device: str):
        """Test error when end_axis <= start_axis"""
        tensor = sparse_1d1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="Expected end_axis to be greater than start_axis",
        ):
            sparse_flatten(tensor, start_axis=1, end_axis=1)

    def test_error_flatten_dense_dims(self, device: str):
        """Test error when trying to flatten dense dimensions"""
        tensor = sparse_1d1d_tensor(device)  # Has 1 sparse dim, 1 dense dim
        with pytest.raises(
            (
                torch.jit.Error,
                NotImplementedError,
            ),  # pyright: ignore[reportArgumentType]
            match="does not currently support flattening dense dims",
        ):
            sparse_flatten(tensor, start_axis=0, end_axis=1)

    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    @given(
        tensor_config=random_sparse_tensor_strategy(max_dim=8),
        axis_strategy=st.data(),
    )
    def test_hypothesis(
        self,
        tensor_config: dict,
        axis_strategy: st.DataObject,
        device: str,
    ):
        """Property-based test for sparse_flatten"""
        # Ensure we have at least 2 sparse dimensions to flatten
        tensor_config["sparse_shape"] = tensor_config["sparse_shape"][
            :4
        ]  # Limit dimensions
        if len(tensor_config["sparse_shape"]) < 2:
            assume(False)

        tensor = random_sparse_tensor(**tensor_config, device=device)

        # Only test flattening sparse dimensions
        max_axis = tensor.sparse_dim() - 1
        if max_axis < 1:
            assume(False)

        start_axis = axis_strategy.draw(st.integers(0, max_axis - 1))
        end_axis = axis_strategy.draw(st.integers(start_axis + 1, max_axis))

        out = sparse_flatten(tensor, start_axis, end_axis)

        # Properties to check:
        # 1. Number of non-zeros unchanged
        assert out._nnz() == tensor._nnz()

        # 2. Values unchanged
        assert torch.equal(out.values(), tensor.values())

        # 3. Coalesced state preserved
        assert out.is_coalesced() == tensor.is_coalesced()

        # 4. Compare against dense version
        dense_flat = tensor.to_dense()
        # Flatten the dense tensor in the same way
        shape = list(tensor.shape)
        new_shape = (
            shape[:start_axis]
            + [math.prod(shape[start_axis : end_axis + 1])]
            + shape[end_axis + 1 :]
        )
        dense_flat = dense_flat.reshape(new_shape)
        assert torch.equal(out.to_dense(), dense_flat)


@pytest.mark.cpu_and_cuda
class TestSparseSqueeze:
    def test_unit_squeeze_sparse_dim(self, device: str):
        """Test squeezing a sparse dimension of size 1"""
        indices = torch.tensor([[0, 0], [0, 1]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(1, 2)).coalesce()

        out = sparse_squeeze(tensor, dim=0)

        assert out.shape == (2,)
        assert torch.equal(out.indices(), torch.tensor([[0, 1]], device=device))
        assert torch.equal(out.values(), values)

    def test_unit_squeeze_dense_dim(self, device: str):
        """Test squeezing a dense dimension of size 1"""
        indices = torch.tensor([[0, 1]], device=device)
        values = torch.tensor([[[1.0]], [[2.0]]], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 1, 1)).coalesce()

        out = sparse_squeeze(tensor, dim=2)  # Squeeze last (dense) dim

        assert out.shape == (2, 1)
        assert torch.equal(out.indices(), indices)
        assert torch.equal(out.values(), values.squeeze(-1))

    def test_unit_squeeze_negative_indexing(self, device: str):
        """Test squeezing with negative index"""
        indices = torch.tensor([[0, 0], [0, 1]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(1, 2)).coalesce()

        out = sparse_squeeze(tensor, dim=-2)  # Same as dim=0

        assert out.shape == (2,)
        assert torch.equal(out.indices(), torch.tensor([[0, 1]], device=device))

    def test_unit_no_squeeze_when_size_not_1(self, device: str):
        """Test that dimension is not squeezed when size != 1"""
        indices = torch.tensor([[0, 1], [0, 1]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2)).coalesce()

        out = sparse_squeeze(tensor, dim=0)

        # Should return same tensor unchanged
        assert out.shape == tensor.shape
        assert torch.equal(out.indices(), tensor.indices())
        assert torch.equal(out.values(), tensor.values())

    def test_unit_squeeze_hybrid_tensor(self, device: str):
        """Test squeezing hybrid sparse tensor"""
        indices = torch.tensor([[0]], device=device)
        values = torch.tensor([[[1.0, 2.0]]], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(1, 1, 2)).coalesce()

        # Squeeze sparse dim
        out1 = sparse_squeeze(tensor, dim=0)
        assert out1.shape == (1, 2)
        assert out1.sparse_dim() == 0

        # Squeeze dense dim
        out2 = sparse_squeeze(tensor, dim=1)
        assert out2.shape == (1, 2)
        assert out2.sparse_dim() == 1

    def test_error_not_sparse(self, device: str):
        """Test error when passed a dense tensor"""
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="Received non-sparse tensor",
        ):
            sparse_squeeze(torch.randn(1, 4, device=device), dim=0)

    def test_error_invalid_dim(self, device: str):
        """Test error with out-of-range dimension"""
        tensor = sparse_1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, IndexError),  # pyright: ignore[reportArgumentType]
            match="out of range",
        ):
            sparse_squeeze(tensor, dim=5)

    def test_error_requires_grad_not_coalesced(self, device: str):
        """Test error when tensor requires grad but is not coalesced"""
        indices = torch.tensor([[0, 0], [0, 0]], device=device)
        values = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        tensor = torch.sparse_coo_tensor(indices, values, size=(1, 2))

        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="must be coalesced",
        ):
            sparse_squeeze(tensor, dim=0)

    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    @given(
        tensor_config=random_sparse_tensor_strategy(max_dim=8),
        dim_to_squeeze=st.data(),
        make_squeezable=st.booleans(),
        use_negative=st.booleans(),
    )
    def test_hypothesis(
        self,
        tensor_config: dict,
        dim_to_squeeze: st.DataObject,
        make_squeezable: bool,
        use_negative: bool,
        device: str,
    ):
        """Property-based test for sparse_squeeze"""
        # Ensure we have at least one dimension
        if not tensor_config["sparse_shape"] and not tensor_config["dense_shape"]:
            tensor_config["sparse_shape"] = [3]

        # Optionally make one dimension size 1
        if make_squeezable:
            all_dims = tensor_config["sparse_shape"] + tensor_config["dense_shape"]
            if all_dims:
                squeeze_idx = random.randint(0, len(all_dims) - 1)
                if squeeze_idx < len(tensor_config["sparse_shape"]):
                    tensor_config["sparse_shape"][squeeze_idx] = 1
                else:
                    tensor_config["dense_shape"][
                        squeeze_idx - len(tensor_config["sparse_shape"])
                    ] = 1

        tensor = random_sparse_tensor(**tensor_config, device=device)
        ndim = tensor.ndim

        if ndim == 0:
            assume(False)

        dim = dim_to_squeeze.draw(st.integers(0, ndim - 1))
        if use_negative:
            dim = dim - ndim

        out = sparse_squeeze(tensor, dim=dim)

        # Properties:
        # 1. If dimension has size 1, it should be removed
        positive_dim = dim if dim >= 0 else ndim + dim
        if tensor.shape[positive_dim] == 1:
            expected_shape = list(tensor.shape)
            expected_shape.pop(positive_dim)
            assert list(out.shape) == expected_shape
        else:
            # Otherwise unchanged
            assert out.shape == tensor.shape

        # 2. Number of non-zeros unchanged
        assert out._nnz() == tensor._nnz()

        # 3. Compare against dense squeeze
        dense_out = torch.squeeze(tensor.to_dense(), dim=dim)
        assert torch.equal(out.to_dense(), dense_out)


@st.composite
def resize_strategy(
    draw: st.DrawFn, max_expand: int = 5, max_dim: Optional[int] = None
):
    tensor_config = draw(random_sparse_tensor_strategy(max_dim=max_dim))
    ndims = len(tensor_config["sparse_shape"]) + len(tensor_config["dense_shape"])
    expand = draw(st.lists(st.integers(0, max_expand), min_size=ndims, max_size=ndims))

    tensor_config.update({"expand_dims": expand})

    return tensor_config


@pytest.mark.cpu_and_cuda
class TestSparseResize:
    def test_unit_resize_2d(self, device: str):
        """Test resizing a 2D sparse tensor to larger shape"""
        indices = torch.tensor([[0, 1], [1, 0]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2))

        out = sparse_resize(tensor, [4, 4])

        assert out.shape == (4, 4)
        assert torch.equal(out.indices(), indices)
        assert torch.equal(out.values(), values)
        # Check that values are in correct positions
        expected = torch.zeros(4, 4, device=device)
        expected[0, 1] = 1.0
        expected[1, 0] = 2.0
        assert torch.equal(out.to_dense(), expected)

    def test_unit_resize_3d_hybrid(self, device: str):
        """Test resizing a 3D hybrid sparse tensor including dense dims"""
        indices = torch.tensor([[0, 1]], device=device)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2)).coalesce()

        out = sparse_resize(tensor, [5, 3])

        assert out.shape == (5, 3)
        assert torch.equal(out.indices(), indices)
        expected_new_values = torch.tensor(
            [[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], device=device
        )
        assert torch.equal(out.values(), expected_new_values)

    def test_unit_resize_same_size(self, device: str):
        """Test resizing to same size"""
        tensor = sparse_1d_tensor(device).coalesce()
        out = sparse_resize(tensor, [4])

        assert out.shape == tensor.shape
        assert torch.equal(out.indices(), tensor.indices())
        assert torch.equal(out.values(), tensor.values())

    def test_error_not_sparse(self, device: str):
        """Test error when passed a dense tensor"""
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="Received non-sparse tensor",
        ):
            sparse_resize(torch.randn(3, 4, device=device), [5, 5])

    def test_error_different_ndim(self, device: str):
        """Test error when new shape has different number of dims"""
        tensor = sparse_1d_tensor(device)
        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="different number of dims",
        ):
            sparse_resize(tensor, [2, 2])

    def test_error_smaller_dimension(self, device: str):
        """Test error when new shape has smaller dimension"""
        indices = torch.tensor([[0, 1], [1, 0]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        tensor = torch.sparse_coo_tensor(indices, values, size=(2, 2))

        with pytest.raises(
            (torch.jit.Error, ValueError),  # pyright: ignore[reportArgumentType]
            match="at least as large",
        ):
            sparse_resize(tensor, [1, 2])  # First dim smaller

    @example(
        tensor_config={
            "sparse_shape": [0],
            "dense_shape": [0, 0],
            "sparsity": 0.0,
            "seed": 0,
            "dtype": torch.float32,
            "expand_dims": [0, 0, 1],
        },
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    @given(tensor_config=resize_strategy(max_dim=10))
    def test_hypothesis_resize(
        self,
        tensor_config: dict,
        device: str,
    ):
        """Property-based test for sparse_resize"""
        tensor = random_sparse_tensor(**tensor_config, device=device)

        # Generate new shape that's at least as large in every dimension
        old_shape = list(tensor.shape)
        if not old_shape:  # Handle 0-dim case
            assume(False)

        # For each dimension, expand by 0 to 5
        expansions = tensor_config["expand_dims"]
        new_shape = [old + exp for old, exp in zip(old_shape, expansions)]

        out: Tensor = sparse_resize(tensor, new_shape)

        # Properties to check:

        # 1. Shape is correct
        assert list(out.shape) == new_shape

        # 2. Number of non-zeros unchanged
        assert out._nnz() == tensor._nnz()

        # 3. Indices unchanged
        assert torch.equal(out.indices(), tensor.indices())

        # 4. Coalesced state preserved
        assert out.is_coalesced() == tensor.is_coalesced()

        # 5. Values have correct shape and content
        old_values = tensor.values()
        new_values = out.values()

        # For sparse-only tensors, values should be unchanged
        if tensor.dense_dim() == 0:
            assert torch.equal(old_values, new_values)
        else:
            # For hybrid tensors, check that old values are contained in new values
            # with zeros prepended in expanded dense dimensions
            sparse_dims = tensor.sparse_dim()
            old_dense_shape = old_shape[sparse_dims:]
            new_dense_shape = new_shape[sparse_dims:]

            # Values should have shape [nnz] + new_dense_shape
            expected_values_shape = [tensor._nnz()] + new_dense_shape
            assert list(new_values.shape) == expected_values_shape

            # Check that the original values are in the correct position
            # (zeros prepended, original values at the end of each dimension)
            if old_dense_shape:
                slices = [slice(None)]  # nnz dimension
                for old_size, new_size in zip(old_dense_shape, new_dense_shape):
                    start = new_size - old_size
                    slices.append(slice(start, new_size))

                extracted = new_values[tuple(slices)]
                assert torch.equal(extracted, old_values)

                # Check that prepended values are zeros
                for i in range(len(old_dense_shape)):
                    dim = i + 1  # +1 for nnz dimension
                    expansion = new_dense_shape[i] - old_dense_shape[i]
                    if expansion > 0:
                        # Check zeros in the prepended part
                        zero_slices = list(slices)
                        zero_slices[dim] = slice(0, expansion)
                        zeros_part = new_values[tuple(zero_slices)]
                        assert torch.all(zeros_part == 0)

        # 6. Compare with manually constructed expected result
        # Create a new sparse tensor with expanded shape and check equality
        expected_sparse = torch.sparse_coo_tensor(
            tensor.indices(),
            out.values(),
            new_shape,
            is_coalesced=tensor.is_coalesced(),
        )

        # The resized tensor should have the same structure
        assert torch.equal(out.indices(), expected_sparse.indices())
        assert torch.equal(out.values(), expected_sparse.values())
        assert out.shape == expected_sparse.shape
