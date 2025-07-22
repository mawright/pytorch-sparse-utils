import math
import torch
import pytest

from pytorch_sparse_utils.indexing.unique import unique_rows
from pytorch_sparse_utils.indexing.scatter import scatter_to_sparse_tensor
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from .. import random_sparse_tensor, random_sparse_tensor_strategy


@pytest.mark.cpu_and_cuda
class TestUniqueRows:
    def test_basic_functionality(self, device):
        tensor = torch.tensor(
            [
                [1, 2, 3],
                [1, 2, 3],
                [7, 8, 9],
                [4, 5, 6],
                [4, 5, 6],
            ],
            device=device,
        )
        unique_inds = unique_rows(tensor)

        assert torch.equal(unique_inds, torch.tensor([0, 2, 3], device=device))

    def test_negative_values(self, device):
        tensor = torch.tensor(
            [
                [1, 2, 3],
                [-1, -2, -3],
                [-1, -2, -3],
                [4, -10, 3],
            ],
            device=device,
        )
        unique_inds = unique_rows(tensor)

        assert torch.equal(unique_inds, torch.tensor([0, 1, 3], device=device))

    def test_sorted(self, device):
        tensor = torch.tensor(
            [[-1, -3, 5], [3, 20, 44], [1, 2, 3], [-1, -3, 5], [3, 20, 44]],
            device=device,
        )

        unique_unsorted = unique_rows(tensor, sorted=False)
        unique_sorted = unique_rows(tensor, sorted=True)

        assert not torch.equal(unique_sorted, unique_unsorted)
        assert torch.equal(unique_sorted, torch.tensor([0, 1, 2], device=device))
        assert torch.equal(unique_unsorted, torch.tensor([0, 2, 1], device=device))

    def test_error_wrong_dim(self, device):
        tensor = torch.randint(0, 100, size=(10,), device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected a 2D tensor",
        ):
            unique_rows(tensor)

    def test_error_not_int(self, device):
        tensor_float = torch.randn(10, 10, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected integer tensor",
        ):
            unique_rows(tensor_float)

        tensor_complex = torch.randn(10, 10, device=device, dtype=torch.complex64)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected integer tensor",
        ):
            unique_rows(tensor_complex)

    def test_error_overflow(self, device):
        tensor = torch.randint(-100, 100, size=(10, 4), device=device, dtype=torch.long)
        tensor[0, :] = torch.iinfo(torch.long).max
        with pytest.raises(
            (OverflowError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Tensor contains values at or near",
        ):
            unique_rows(tensor)

        tensor[0, :] = torch.iinfo(torch.long).max - 100
        with pytest.raises(
            (OverflowError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="would cause integer overflow",
        ):
            unique_rows(tensor)


@pytest.mark.cpu_and_cuda
class TestScatterToSparseTensor:
    def test_update_existing_values(self, device):
        """Test updating existing values in sparse tensor."""
        indices = torch.tensor([[0, 1, 2], [0, 1, 0]], device=device)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3, 2), device=device)

        # Update first and second values
        update_indices = torch.tensor([[0, 0], [1, 1]], device=device)
        new_values = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device=device)

        result = scatter_to_sparse_tensor(sparse, update_indices, new_values)

        expected_values = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0], [5.0, 6.0]], device=device
        )
        assert torch.equal(result.coalesce().values(), expected_values)

    def test_add_new_values(self, device):
        """Test adding new values to sparse tensor."""
        indices = torch.tensor([[0, 1], [0, 1]], device=device)
        values = torch.tensor([[1.0], [2.0]], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2, 1), device=device)

        # Add value at new location [0, 1]
        new_indices = torch.tensor([[0, 1]], device=device)
        new_values = torch.tensor([[3.0]], device=device)

        result = scatter_to_sparse_tensor(sparse, new_indices, new_values)

        # Check that new value was added
        dense = result.to_dense()
        assert dense[0, 1, 0] == 3.0
        assert result._nnz() == 3

    def test_duplicate_indices(self, device):
        """Test handling of duplicate indices (last write wins)."""
        indices = torch.tensor([[0, 1], [0, 0]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2), device=device)

        # Update same location twice
        update_indices = torch.tensor([[0, 0], [0, 0]], device=device)
        update_values = torch.tensor([10.0, 20.0], device=device)

        result = scatter_to_sparse_tensor(sparse, update_indices, update_values)

        # Should be one of the two values (not added)
        value = result.values()[0]
        assert value == 10.0 or value == 20.0

    def test_check_all_specified_valid(self, device):
        """Test check_all_specified with valid indices."""
        indices = torch.tensor([[0, 1], [0, 1]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2), device=device)

        # Update only existing indices
        update_indices = torch.tensor([[0, 0], [1, 1]], device=device)
        update_values = torch.tensor([10.0, 20.0], device=device)

        # Should not raise
        result = scatter_to_sparse_tensor(
            sparse, update_indices, update_values, check_all_specified=True
        )
        assert result.coalesce().values()[0] == 10.0
        assert result.coalesce().values()[1] == 20.0

    def test_check_all_specified_invalid(self, device):
        """Test check_all_specified raises with invalid indices."""
        indices = torch.tensor([[0, 1], [0, 1]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2), device=device)

        # Try to update non-existing index
        update_indices = torch.tensor([[0, 1]], device=device)  # [0, 1] doesn't exist
        update_values = torch.tensor([10.0], device=device)

        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="check_all_specified",
        ):
            scatter_to_sparse_tensor(
                sparse, update_indices, update_values, check_all_specified=True
            )

    def test_empty_sparse_tensor(self, device):
        """Test scattering to empty sparse tensor."""
        sparse = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty(0, 3, device=device),
            (4, 4, 3),
        )

        # Add values to empty tensor
        indices = torch.tensor([[0, 0], [2, 3]], device=device)
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

        result = scatter_to_sparse_tensor(sparse, indices, values)

        assert result._nnz() == 2
        assert torch.equal(result.coalesce().values(), values)

    def test_gradient_preservation(self, device):
        """Test that gradients are preserved."""
        indices = torch.tensor([[0, 1], [0, 1]], device=device)
        values = torch.tensor([1.0, 2.0], requires_grad=True)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2), device=device)

        update_indices = torch.tensor([[0, 0]], device=device)
        update_values = torch.tensor([3.0], requires_grad=True, device=device)

        result = scatter_to_sparse_tensor(sparse, update_indices, update_values)

        assert result.values().requires_grad

        # Test backward pass
        loss = result.values().sum()
        loss.backward()
        assert update_values.grad is not None

    def test_different_dtypes(self, device):
        """Test with different tensor dtypes."""
        # Integer tensor
        indices = torch.tensor([[0, 1], [0, 1]], device=device)
        values = torch.tensor([10, 20], dtype=torch.int32, device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2))

        update_indices = torch.tensor([[0, 0]], device=device)
        update_values = torch.tensor([30], dtype=torch.int32, device=device)

        result = scatter_to_sparse_tensor(sparse, update_indices, update_values)
        assert result.dtype == torch.int32
        assert result.values()[0] == 30

    def test_batch_dimensions(self, device):
        """Test with batch dimensions in index and value tensors."""
        indices = torch.tensor([[0, 1, 2], [0, 1, 0]], device=device)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3, 2), device=device)

        sparse_2 = sparse.clone()

        # Batch of 2x2 updates
        batch_indices = torch.tensor(
            [[[0, 0], [1, 1]], [[2, 0], [0, 2]]], device=device
        )
        batch_values = torch.tensor(
            [[[10.0, 11.0], [12.0, 13.0]], [[14.0, 15.0], [16.0, 17.0]]], device=device
        )

        result = scatter_to_sparse_tensor(sparse, batch_indices, batch_values)

        # Check some updated values
        dense = result.to_dense()
        assert torch.equal(dense[0, 0], torch.tensor([10.0, 11.0], device=device))
        assert torch.equal(dense[1, 1], torch.tensor([12.0, 13.0], device=device))
        # New value added
        assert torch.equal(dense[0, 2], torch.tensor([16.0, 17.0], device=device))

        # Try with flattened batch dims to ensure it's the same
        flat_indices = batch_indices.reshape(-1, 2)
        flat_values = batch_values.reshape(-1, 2)

        result_flat = scatter_to_sparse_tensor(sparse_2, flat_indices, flat_values)

        assert torch.equal(result.indices(), result_flat.indices())
        assert torch.equal(result.values(), result_flat.values())

    def test_error_nonmatching_batch_dims(self, device):
        """Test error when batch dims for index_tensor and values don't match"""
        indices = torch.tensor([[0, 1, 2], [0, 1, 0]], device=device)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3, 2), device=device)

        batch_indices = torch.tensor(
            [[[0, 0], [1, 1]], [[2, 0], [0, 2]]], device=device
        )
        batch_values = torch.randn(3, 2, 2, device=device)

        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected matching batch dims",
        ):
            scatter_to_sparse_tensor(sparse, batch_indices, batch_values)


@pytest.mark.cpu_and_cuda
class TestScatterToSparseTensorHypothesis:
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        sparse_params=random_sparse_tensor_strategy(max_dim=32),
        scatter_batch_dims=st.lists(st.integers(1, 20), min_size=1, max_size=3),
    )
    def test_scatter_sparse_vs_dense(self, sparse_params, scatter_batch_dims, device):
        """Property-based test comparing sparse scatter against dense equivalent."""
        # Assume sparse tensor has more than 0 elements
        assume(math.prod(sparse_params["sparse_shape"]) > 0)

        sparse_tensor = random_sparse_tensor(
            sparse_shape=sparse_params["sparse_shape"],
            dense_shape=sparse_params["dense_shape"],
            sparsity=sparse_params["sparsity"],
            seed=sparse_params["seed"],
            device=device,
            dtype=sparse_params["dtype"],
        )
        seed = sparse_params["seed"]
        rng = torch.Generator(device)
        rng.manual_seed(seed)

        # Convert to dense for comparison
        dense_tensor = sparse_tensor.to_dense().clone()

        # Create valid indices within the sparse shape bounds
        index_tensor = torch.stack(
            [
                torch.randint(
                    0, dim_size, scatter_batch_dims, device=device, generator=rng
                )
                for dim_size in sparse_params["sparse_shape"]
            ],
            dim=-1,
        )

        # Generate values to scatter
        value_shape = scatter_batch_dims + sparse_params["dense_shape"]
        values = torch.empty(
            value_shape, device=device, dtype=sparse_params["dtype"]
        )
        if torch.is_floating_point(values):
            values.normal_(generator=rng)
        else:
            values.random_(generator=rng)

        # Perform scatter on sparse tensor
        result_sparse = scatter_to_sparse_tensor(sparse_tensor, index_tensor, values)

        # Perform equivalent operation on dense tensor
        # Convert index_tensor to a format suitable for indexing
        dense_tensor[index_tensor.unbind(-1)] = values

        # Compare results
        result_dense = result_sparse.to_dense()

        # Check shapes match
        assert result_dense.shape == dense_tensor.shape

        # Check values match
        indices_flat = index_tensor.reshape(-1, result_sparse.sparse_dim())
        if unique_rows(indices_flat).shape[0] == indices_flat.shape[0]:
            # No duplicate indices so should be deterministic writes
            assert torch.equal(result_dense, dense_tensor)
