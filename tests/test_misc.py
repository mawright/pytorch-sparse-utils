import pytest
import torch
from torch import Tensor

from pytorch_sparse_utils.misc import prod, unpack_sparse_tensors, _pytorch_atleast_2_5
from pytorch_sparse_utils.validation import (
    validate_atleast_nd,
    validate_dim_size,
    validate_nd,
)
from . import random_sparse_tensor


@pytest.mark.cpu_and_cuda
class TestValidate:
    def test_validate_nd(self, device):
        tensor = torch.randn(4, 5, 6, device=device)
        validate_nd(tensor, 3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to be 4D",
        ):
            validate_nd(tensor, 4)

    def test_validate_at_least_nd(self, device):
        tensor = torch.randn(4, 5, 6, device=device)
        validate_atleast_nd(tensor, 3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to have at least",
        ):
            validate_atleast_nd(tensor, 4)

    def test_validate_dim_size(self, device):
        tensor = torch.randn(3, 4, 5, device=device)
        validate_dim_size(tensor, dim=0, expected_size=3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match=r"Expected tensor to have shape\[0\]=4",
        ):
            validate_dim_size(tensor, dim=0, expected_size=4)


@pytest.mark.cpu_and_cuda
def test_prod(device):
    test_list = [1, 2, 3]
    result_list = prod(test_list)
    assert result_list == 6
    assert isinstance(result_list, int)

    test_tensor = torch.tensor(test_list, device=device)
    result_tensor = prod(test_tensor)
    assert isinstance(result_tensor, Tensor)
    assert result_tensor == 6


@pytest.mark.cpu_and_cuda
def test_unpack_sparse_tensors(device):
    sparse_tensor = random_sparse_tensor(
        [8, 16, 16], [4, 32], 0.5, seed=0, device=device
    ).coalesce()

    batch_dict = {
        "X_indices": sparse_tensor.indices(),
        "X_values": sparse_tensor.values(),
        "X_shape": torch.tensor(sparse_tensor.shape, device=device),
    }

    if _pytorch_atleast_2_5:
        with pytest.warns(DeprecationWarning, match="is no longer needed"):
            unpacked = unpack_sparse_tensors(batch_dict)
    else:
        unpacked = unpack_sparse_tensors(batch_dict)

    X = unpacked["X"]
    assert torch.equal(X.indices(), sparse_tensor.indices())
    assert torch.equal(X.values(), sparse_tensor.values())
    assert X.shape == sparse_tensor.shape
