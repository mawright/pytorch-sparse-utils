import numpy as np
import pytest
import sparse
import torch

from pytorch_sparse_utils.conversion import (
    minkowski_to_torch_sparse,
    pydata_sparse_to_torch_sparse,
    spconv_to_torch_sparse,
    torch_sparse_to_minkowski,
    torch_sparse_to_pydata_sparse,
    torch_sparse_to_spconv,
)
from pytorch_sparse_utils.imports import (
    has_minkowskiengine,
    has_spconv,
    ME,
    spconv,
)


class TestTorchSparseToPydataSparse:
    def test_basic_conversion(self):
        # Create a simple sparse tensor
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
        values = torch.tensor([3.0, 4.0, 5.0])
        shape = (2, 3)
        tensor = torch.sparse_coo_tensor(indices, values, shape)

        result = torch_sparse_to_pydata_sparse(tensor)

        assert isinstance(result, sparse.COO)
        assert result.shape == shape
        assert np.array_equal(result.data, values.numpy())
        assert np.array_equal(result.coords, indices.numpy())

    def test_with_zero_values(self):
        # Test with some zero values that should be filtered out
        indices = torch.tensor([[0, 1, 1, 2], [2, 0, 2, 1]])
        values = torch.tensor([3.0, 0.0, 5.0, 0.0])  # Two zeros
        shape = (3, 3)
        tensor = torch.sparse_coo_tensor(indices, values, shape)

        result = torch_sparse_to_pydata_sparse(tensor)

        # Should only have non-zero values
        assert len(result.data) == 2
        assert np.array_equal(result.data, [3, 5])

    def test_requires_grad(self):
        # Test with tensor that requires gradient
        indices = torch.tensor([[0, 1], [2, 0]])
        values = torch.tensor([3.0, 4.0])
        tensor = torch.sparse_coo_tensor(indices, values, (2, 3), requires_grad=True)

        result = torch_sparse_to_pydata_sparse(tensor)

        assert isinstance(result, sparse.COO)
        assert np.array_equal(result.data, [3, 4])

    def test_non_sparse_tensor_raises(self):
        # Should raise assertion error for dense tensor
        tensor = torch.randn(3, 4)

        with pytest.raises(AssertionError):
            torch_sparse_to_pydata_sparse(tensor)

    def test_empty_sparse_tensor(self):
        # Test with empty sparse tensor
        indices = torch.tensor([[], []], dtype=torch.long)
        values = torch.tensor([], dtype=torch.float)
        shape = (3, 3)
        tensor = torch.sparse_coo_tensor(indices, values, shape)

        result = torch_sparse_to_pydata_sparse(tensor)

        assert result.shape == shape
        assert len(result.data) == 0


class TestPydataSparseToTorchSparse:
    def test_basic_conversion(self):
        # Create a simple sparse COO array
        coords = np.array([[0, 1, 1], [2, 0, 2]])
        data = np.array([3.0, 4.0, 5.0])
        shape = (2, 3)
        sparse_array = sparse.COO(coords, data, shape)

        result = pydata_sparse_to_torch_sparse(sparse_array)

        assert isinstance(result, torch.Tensor)
        assert result.is_sparse
        assert result.shape == shape
        assert torch.allclose(result.values(), torch.tensor(data))

    def test_with_device(self):
        # Test device placement
        coords = np.array([[0, 1], [2, 0]])
        data = np.array([3.0, 4.0])
        sparse_array = sparse.COO(coords, data, (2, 3))

        if torch.cuda.is_available():
            result = pydata_sparse_to_torch_sparse(sparse_array, device="cuda")
            assert result.device.type == "cuda"

        result_cpu = pydata_sparse_to_torch_sparse(sparse_array, device="cpu")
        assert result_cpu.device.type == "cpu"

    def test_empty_sparse_array(self):
        # Test with empty sparse array
        sparse_array = sparse.COO(np.array([[], []]), np.array([]), shape=(3, 3))

        result = pydata_sparse_to_torch_sparse(sparse_array)

        assert result.shape == (3, 3)
        assert result.values().numel() == 0

    def test_large_sparse_array(self):
        # Test with larger sparse array
        shape = (100, 100)
        # Create random sparse data
        nnz = 50
        row_indices = np.random.randint(0, shape[0], size=nnz)
        col_indices = np.random.randint(0, shape[1], size=nnz)
        coords = np.array([row_indices, col_indices])
        data = np.random.randn(nnz)
        sparse_array = sparse.COO(coords, data, shape)

        result = pydata_sparse_to_torch_sparse(sparse_array)

        assert result.shape == shape
        assert result.values().shape[0] <= nnz  # May be less due to coalescing


@pytest.mark.skipif(not has_minkowskiengine, reason="MinkowskiEngine not installed")
class TestMinkowskiEngineReal:
    def test_torch_sparse_to_minkowski_real(self):
        # Create sparse tensor with 3D coordinates (batch, x, y) and features
        indices = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [3, 4, 5, 6]], dtype=torch.int
        )
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shape = (2, 2, 2, 10)  # batch_size=2, spatial=(2,2), features=10
        tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()

        result = torch_sparse_to_minkowski(tensor)

        assert hasattr(result, "F")  # Features
        assert hasattr(result, "C")  # Coordinates
        assert result.F.shape[0] == 4  # Number of points
        assert result.F.shape[1] == 1  # Feature dimension

    def test_minkowski_to_torch_sparse_real(self):
        # Create a MinkowskiEngine SparseTensor
        coordinates = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=torch.int
        )
        features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        sparse_tensor = ME.SparseTensor(features, coordinates)

        result = minkowski_to_torch_sparse(sparse_tensor)

        assert isinstance(result, torch.Tensor)
        assert result.is_sparse
        assert result.values().numel() == 4

    def test_roundtrip_minkowski(self):
        # Test roundtrip conversion
        indices = torch.tensor(
            [[0, 0, 1], [0, 1, 0], [0, 1, 1], [2, 3, 4]], dtype=torch.int
        )
        values = torch.tensor([1.0, 2.0, 3.0])
        shape = (2, 2, 2, 5)
        tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()

        me_tensor = torch_sparse_to_minkowski(tensor)
        back_to_torch = minkowski_to_torch_sparse(
            me_tensor, full_scale_spatial_shape=[2, 2]
        )

        # Check that we get back similar structure (may not be identical due to coordinate handling)
        assert back_to_torch.is_sparse
        assert back_to_torch.shape[0] == 2  # batch size preserved


@pytest.mark.skipif(not has_spconv, reason="spconv not installed")
class TestSpconv:
    def test_torch_sparse_to_spconv(self):
        # Create 4D sparse tensor (batch, height, width, features)
        indices = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1]]).T
        values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        shape = (2, 3, 3, 2)  # batch_size=2, spatial_shape=[3,3], features=2
        tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()

        result = torch_sparse_to_spconv(tensor)

        assert isinstance(result, spconv.SparseConvTensor)
        assert result.spatial_shape == [3, 3]
        assert result.batch_size == 2

    def test_spconv_to_torch_sparse(self):
        # Create a SparseConvTensor
        features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        indices = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.int
        ).T
        spatial_shape = [2, 2]
        batch_size = 2
        sparse_conv_tensor = spconv.SparseConvTensor(
            features, indices, spatial_shape, batch_size
        )

        result = spconv_to_torch_sparse(sparse_conv_tensor)

        assert isinstance(result, torch.Tensor)
        assert result.is_sparse
        assert result.shape == (2, 2, 2, 1)  # (batch, h, w, features)

    def test_spconv_squeeze(self):
        # Test squeeze functionality real spconv
        features = torch.tensor([[1.0], [2.0], [3.0]])
        indices = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.int).T
        spatial_shape = [2, 2]
        batch_size = 2
        sparse_conv_tensor = spconv.SparseConvTensor(
            features, indices, spatial_shape, batch_size
        )

        result = spconv_to_torch_sparse(sparse_conv_tensor, squeeze=True)

        assert result.shape == (2, 2, 2)  # Feature dimension squeezed
        assert result.values().shape == (3,)  # Values are 1D

    def test_spconv_squeeze_error(self):
        # Test squeeze functionality real spconv
        features = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        indices = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.int).T
        spatial_shape = [2, 2]
        batch_size = 2
        sparse_conv_tensor = spconv.SparseConvTensor(
            features, indices, spatial_shape, batch_size
        )

        with pytest.raises(ValueError, match="Got `squeeze`=True, but"):
            _ = spconv_to_torch_sparse(sparse_conv_tensor, squeeze=True)


    def test_roundtrip_spconv(self):
        # Test roundtrip conversion
        indices = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1]]).T
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shape = (2, 2, 2)
        tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()

        spconv_tensor = torch_sparse_to_spconv(tensor)
        back_to_torch = spconv_to_torch_sparse(spconv_tensor, squeeze=True)

        assert back_to_torch.is_sparse
        assert back_to_torch.shape == shape
        assert torch.equal(tensor.indices(), back_to_torch.indices())
        assert torch.equal(tensor.values(), back_to_torch.values())

    def test_already_spconv(self):
        #  Test no-op
        features = torch.tensor([[1.0], [2.0], [3.0]])
        indices = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.int).T
        spatial_shape = [2, 2]
        batch_size = 2
        sparse_conv_tensor = spconv.SparseConvTensor(
            features, indices, spatial_shape, batch_size
        )

        result = torch_sparse_to_spconv(
            sparse_conv_tensor  # pyright: ignore[reportArgumentType]
        )
        assert result is sparse_conv_tensor

    def test_already_torch_sparse(self):
        # Test no-op (already torch sparse)
        indices = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1]]).T
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shape = (2, 2, 2)
        tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()

        result = spconv_to_torch_sparse(tensor)
        assert result is tensor


# Integration tests for round-trip conversions
class TestRoundTripConversions:
    def test_torch_pydata_torch_roundtrip(self):
        # Create original sparse tensor
        indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
        values = torch.tensor([3.5, 4.2, 5.1])
        shape = (3, 3)
        original = torch.sparse_coo_tensor(indices, values, shape)

        # Convert to pydata and back
        pydata_array = torch_sparse_to_pydata_sparse(original)
        reconstructed = pydata_sparse_to_torch_sparse(pydata_array)

        # Compare
        assert reconstructed.shape == original.shape
        assert torch.allclose(reconstructed.to_dense(), original.to_dense())
