import torch
import pytest

from pytorch_sparse_utils.indexing.unique import unique_rows
from pytorch_sparse_utils.indexing.scatter import scatter_to_sparse_tensor

@pytest.mark.cpu_and_cuda
class TestUniqueRows:
    def test_basic_functionality(self, device):
        tensor = torch.tensor([
            [1, 2, 3],
            [1, 2, 3],
            [7, 8, 9],
            [4, 5, 6],
            [4, 5, 6],
        ], device=device
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
            ], device=device
        )
        unique_inds = unique_rows(tensor)

        assert torch.equal(unique_inds, torch.tensor([0, 1, 3], device=device))

    def test_sorted(self, device):
        tensor = torch.tensor([
            [-1, -3, 5],
            [3, 20, 44],
            [1, 2, 3],
            [-1, -3, 5],
            [3, 20, 44]
        ], device=device)

        unique_unsorted = unique_rows(tensor, sorted=False)
        unique_sorted = unique_rows(tensor, sorted=True)

        assert not torch.equal(unique_sorted, unique_unsorted)
        assert torch.equal(unique_sorted, torch.tensor([0, 1, 2], device=device))
        assert torch.equal(unique_unsorted, torch.tensor([0, 2, 1], device=device))

    def test_error_wrong_dim(self, device):
        tensor = torch.randint(0, 100, size=(10,), device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected a 2D tensor"
        ):
            unique_rows(tensor)

    def test_error_not_int(self, device):
        tensor_float = torch.randn(10, 10, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected integer tensor"
        ):
            unique_rows(tensor_float)

        tensor_complex = torch.randn(10, 10, device=device, dtype=torch.complex64)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected integer tensor"
        ):
            unique_rows(tensor_complex)

    def test_error_overflow(self, device):
        tensor = torch.randint(-100, 100, size=(10, 4), device=device, dtype=torch.long)
        tensor[0, :] = torch.iinfo(torch.long).max
        with pytest.raises(
            (OverflowError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Tensor contains values at or near"
        ):
            unique_rows(tensor)

        tensor[0, :] = torch.iinfo(torch.long).max - 100
        with pytest.raises(
            (OverflowError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="would cause integer overflow"
        ):
            unique_rows(tensor)
