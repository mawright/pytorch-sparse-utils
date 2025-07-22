import pytest
import torch

from pytorch_sparse_utils.validation import validate_nd, validate_dim_size, validate_atleast_nd

@pytest.mark.cpu_and_cuda
class TestValidate:
    def test_validate_nd(self, device):
        tensor = torch.randn(4, 5, 6, device=device)
        validate_nd(tensor, 3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to be 4D"
        ):
            validate_nd(tensor, 4)

    def test_validate_at_least_nd(self, device):
        tensor = torch.randn(4, 5, 6, device=device)
        validate_atleast_nd(tensor, 3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to have at least"
        ):
            validate_atleast_nd(tensor, 4)

    def test_validate_dim_size(self, device):
        tensor = torch.randn(3, 4, 5, device=device)
        validate_dim_size(tensor, dim=0, expected_size=3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match=r"Expected tensor to have shape\[0\]=4"
        ):
            validate_dim_size(tensor, dim=0, expected_size=4)
