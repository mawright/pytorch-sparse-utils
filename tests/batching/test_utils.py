import pytest
import torch
from torch import Tensor

from pytorch_sparse_utils.batching.batch_utils import (
    batch_dim_to_leading_index,
    batch_indices_to_offsets,
    batch_offsets_from_sparse_tensor_indices,
    batch_offsets_to_indices,
    batch_offsets_to_seq_lengths,
    concatenated_to_padded,
    concatenated_to_sparse_tensor,
    normalize_batch_offsets,
    padded_to_concatenated,
    padded_to_sparse_tensor,
    seq_lengths_to_batch_offsets,
    seq_lengths_to_indices,
    sparse_tensor_to_concatenated,
    sparse_tensor_to_padded,
    split_batch_concatenated_tensor,
)
from pytorch_sparse_utils.validation import validate_atleast_nd


@pytest.fixture
def simple_tensor(device):
    """Create a simple tensor for testing."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=device)


@pytest.fixture
def simple_batch_offsets(device):
    """Create simple batch offsets for testing."""
    return torch.tensor([0, 3, 7, 9], device=device)


@pytest.mark.cpu_and_cuda
class TestNormalizeBatchOffsets:
    def test_already_normalized(self, device):
        """Test when offsets already end with total length."""
        batch_offsets = torch.tensor([0, 5, 10], device=device)
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        assert torch.equal(result, batch_offsets)

    def test_needs_normalizing(self, device):
        """Test when normalization is needed."""
        batch_offsets = torch.tensor([5, 8], device=device)
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        expected = torch.tensor([0, 5, 8, 10], device=device)
        assert torch.equal(result, expected)

    def test_different_dtype(self, device):
        """Test with different dtypes."""
        batch_offsets = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        assert result.dtype == batch_offsets.dtype

    def test_float_dtype_error(self, device):
        """Test with non-integer dtype."""
        batch_offsets = torch.tensor([0, 5, 8], dtype=torch.float, device=device)
        total_length = 10
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected integer tensor",
        ):
            _ = normalize_batch_offsets(batch_offsets, total_length)

    def test_total_length_error(self, device):
        """Test with total_length too small."""
        batch_offsets = torch.tensor([0, 5, 8], device=device)
        total_length = 6
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Max value of batch_offsets",
        ):
            _ = normalize_batch_offsets(batch_offsets, total_length)

    def test_list(self):
        """Test with list input"""
        batch_offsets = [5, 8]
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        expected = [0, 5, 8, 10]
        assert len(result) == len(expected)
        assert isinstance(result, type(expected))
        assert all(result[i] == expected[i] for i in range(len(expected)))

    def test_list_already_normalized(self):
        """Test with already-normalized list"""
        batch_offsets = [0, 5, 8, 10]
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        assert result == batch_offsets

    def test_list_total_length_error(self):
        """Test list with bad total_length"""
        batch_offsets = [0, 5, 8, 10]
        total_length = 6
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Max value of batch_offsets",
        ):
            _ = normalize_batch_offsets(batch_offsets, total_length)


@pytest.mark.cpu_and_cuda
class TestSeqLengthsToBatchOffsets:
    def test_basic_functionality(self, device):
        """Test basic functionality of seq_lenghts_to_batch_offsets."""
        seq_lengths = torch.tensor([2, 4, 6, 43, 3], device=device)
        result = seq_lengths_to_batch_offsets(seq_lengths)
        expected = torch.tensor([0, 2, 6, 12, 55, 58], device=device)
        assert torch.equal(result, expected)

    def test_empty_tensor(self, device):
        """Test with empty tensor."""
        seq_lengths = torch.tensor([], device=device)
        result = seq_lengths_to_batch_offsets(seq_lengths)
        expected = torch.tensor([0], device=device)
        assert torch.equal(result, expected)

    def test_list(self):
        """Test with list input"""
        seq_lengths = [2, 4, 6, 43, 3]
        result = seq_lengths_to_batch_offsets(seq_lengths)
        expected = [0, 2, 6, 12, 55, 58]
        assert len(result) == len(expected)
        assert isinstance(result, type(expected))
        assert all(result[i] == expected[i] for i in range(len(result)))


@pytest.mark.cpu_and_cuda
class TestBatchOffsetsToSeqLengths:
    def test_basic_functionality(self, device):
        """Test basic functionality of batch_offsets_to_seq_lengths."""
        batch_offsets = torch.tensor([0, 3, 7, 10], device=device)
        result = batch_offsets_to_seq_lengths(batch_offsets)
        expected = torch.tensor([3, 4, 3], device=device)
        assert torch.equal(result, expected)

    def test_single_batch(self, device):
        """Test with a single batch."""
        batch_offsets = torch.tensor([0, 5], device=device)
        result = batch_offsets_to_seq_lengths(batch_offsets)
        expected = torch.tensor([5], device=device)
        assert torch.equal(result, expected)

    def test_list(self):
        """Test with list input."""
        batch_offsets = [0, 3, 7, 10]
        result = batch_offsets_to_seq_lengths(batch_offsets)
        expected = [3, 4, 3]
        assert len(result) == len(expected)
        assert isinstance(result, type(expected))
        assert all(result[i] == expected[i] for i in range(len(result)))


@pytest.mark.cpu_and_cuda
class TestSeqLengthsToIndices:
    def test_basic_functionality(self, device):
        """Test basic functionality of seq_lengths_to_indices."""
        seq_lengths = torch.tensor([5, 4], device=device)
        result = seq_lengths_to_indices(seq_lengths)
        expected = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], device=device)
        assert torch.equal(result, expected)

    def test_single_batch(self, device):
        """Test with a single batch"""
        seq_lengths = torch.tensor([3], device=device)
        result = seq_lengths_to_indices(seq_lengths)
        expected = torch.tensor([0, 0, 0], device=device)
        assert torch.equal(result, expected)

    def test_empty(self, device):
        """Test with an empty tensor"""
        seq_lengths = torch.empty([0], device=device, dtype=torch.long)
        result = seq_lengths_to_indices(seq_lengths)
        expected = torch.empty([0], device=device, dtype=torch.long)
        assert torch.equal(result, expected)

    def test_scalar(self, device):
        """Test with a scalar value."""
        seq_lengths = torch.tensor(4, device=device)
        result = seq_lengths_to_indices(seq_lengths)
        expected = torch.tensor([0, 0, 0, 0], device=device)
        assert torch.equal(result, expected)


@pytest.mark.cpu_and_cuda
class TestValidateTensorDims:
    def test_valid_dims(self, device):
        """Test with valid dimensions."""
        tensor = torch.rand(3, 4, 5, device=device)
        # Should not raise any exception
        validate_atleast_nd(tensor, 3)

    def test_too_few_dims(self, device):
        """Test with too few dimensions."""
        tensor = torch.rand(3, 4, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected tensor to have at least 3 dimensions",
        ):
            validate_atleast_nd(tensor, 3)

    def test_custom_name(self, device):
        """Test with custom tensor name."""
        tensor = torch.rand(3, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected custom_tensor to have at least 2 dimensions",
        ):
            validate_atleast_nd(tensor, 2, name="custom_tensor")


@pytest.mark.cpu_and_cuda
class TestBatchOffsetsToIndices:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        batch_offsets = torch.tensor([0, 3, 7, 10], device=device)
        result = batch_offsets_to_indices(batch_offsets)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device=device)
        assert torch.equal(result, expected)

    def test_with_total_seq_length(self, device):
        """Test with total_seq_length parameter."""
        batch_offsets = torch.tensor([0, 3, 7], device=device)
        total_seq_length = 10
        result = batch_offsets_to_indices(batch_offsets, total_seq_length)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device=device)
        assert torch.equal(result, expected)

    def test_list_input(self, device):
        """Test with list batch offsets"""
        batch_offsets = [0, 3, 7, 10]
        result = batch_offsets_to_indices(batch_offsets, device=device)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device=device)
        assert torch.equal(result, expected)


@pytest.mark.cpu_and_cuda
class TestBatchIndicesToOffsets:
    def test_basic_functionality(self, device):
        """Test basic functionality with batch_offsets_to_indices"""
        batch_offsets = torch.tensor([0, 3, 7, 10], device=device)
        indices = batch_offsets_to_indices(batch_offsets)
        offsets_2 = batch_indices_to_offsets(indices)
        assert torch.equal(batch_offsets, offsets_2)

    def test_empty(self, device):
        """Test with empty tensor"""
        indices = torch.tensor([], device=device, dtype=torch.long)
        offsets = batch_indices_to_offsets(indices)
        assert torch.equal(
            offsets, torch.zeros(1, device=indices.device, dtype=torch.long)
        )


@pytest.mark.cpu_and_cuda
class TestSplitBatchConcattedTensor:
    def test_basic_functionality(
        self, simple_tensor: Tensor, simple_batch_offsets: Tensor, device
    ):
        """Test basic splitting functionality."""
        result = split_batch_concatenated_tensor(simple_tensor, simple_batch_offsets)

        # Expected splits: [1,2,3], [4,5,6,7], [8,9]
        assert len(result) == 3
        assert torch.equal(result[0], torch.tensor([1.0, 2.0, 3.0], device=device))
        assert torch.equal(result[1], torch.tensor([4.0, 5.0, 6.0, 7.0], device=device))
        assert torch.equal(result[2], torch.tensor([8.0, 9.0], device=device))

    def test_multidimensional_tensor(self, device):
        """Test with multidimensional tensors."""
        tensor = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
                [13.0, 14.0],
            ],
            device=device,
        )

        batch_offsets = torch.tensor([0, 3, 5, 7], device=device)
        result = split_batch_concatenated_tensor(tensor, batch_offsets)

        assert len(result) == 3
        assert torch.equal(result[0], tensor[0:3])
        assert torch.equal(result[1], tensor[3:5])
        assert torch.equal(result[2], tensor[5:7])


@pytest.mark.cpu_and_cuda
class TestConcatenatedToPadded:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            device=device,
        )

        batch_offsets = torch.tensor([0, 3, 6], device=device)

        result, padding_mask = concatenated_to_padded(tensor, batch_offsets)

        # Expected shape: [2, 3, 2] (2 batches, max_len=3, feature_dim=2)
        assert result.shape == (2, 3, 2)

        # Check values
        assert torch.equal(result[0], tensor[0:3])
        assert torch.equal(result[1], tensor[3:6])

        # No padding needed in this case
        expected_mask = torch.zeros(2, 3, dtype=torch.bool, device=device)
        assert torch.equal(padding_mask, expected_mask)

    def test_with_padding(self, device):
        """Test with sequences of different lengths requiring padding."""
        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device=device
        )

        batch_offsets = torch.tensor([0, 2, 5], device=device)

        result, padding_mask = concatenated_to_padded(tensor, batch_offsets)

        # Expected shape: [2, 3, 2] (2 batches, max_len=3, feature_dim=2)
        assert result.shape == (2, 3, 2)

        # Check padding mask
        expected_mask = torch.tensor(
            [[False, False, True], [False, False, False]],
            dtype=torch.bool,
            device=device,
        )
        assert torch.equal(padding_mask, expected_mask)

        # Check padded values
        assert torch.equal(result[0, 0:2], tensor[0:2])
        assert torch.all(result[0, 2] == 0.0)  # Padded with zeros

    def test_custom_pad_value(self, device):
        """Test with custom pad value."""
        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device=device
        )

        batch_offsets = torch.tensor([0, 1, 4], device=device)

        result, _ = concatenated_to_padded(tensor, batch_offsets, pad_value=-1.0)

        # Check padded values have custom pad value
        assert torch.all(result[0, 1:3] == -1.0)

    def test_scalar_feature(self, device):
        """Test with a 1D tensor (scalar feature)"""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        batch_offsets = torch.tensor([0, 1, 3, 4, 5], device=device)

        result, pad_mask = concatenated_to_padded(tensor, batch_offsets)

        assert result.shape == (4, 2)

        expected_result = torch.tensor(
            [[1.0, 0.0], [2.0, 3.0], [4.0, 0.0], [5.0, 0.0]], device=device
        )
        expected_mask = torch.tensor(
            [[False, True], [False, False], [False, True], [False, True]], device=device
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(pad_mask, expected_mask)

    def test_error_handling(self, device):
        """Test error handling."""
        # Test with scalar tensor
        tensor_1d = torch.tensor(1.0, device=device)
        batch_offsets = torch.tensor([0, 1], device=device)

        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected tensor to have at least 1",
        ):
            concatenated_to_padded(tensor_1d, batch_offsets)

        # Test with bad batch_offsets
        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device=device
        )
        batch_offsets_2d = torch.tensor([[0, 3]], device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected batch_offsets to be 1D",
        ):
            _ = concatenated_to_padded(tensor, batch_offsets_2d)


@pytest.mark.cpu_and_cuda
class TestPaddedToConcatenated:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        # Create a batched tensor with padding
        tensor = torch.tensor(
            [
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
                [[4.0, 4.1], [5.0, 5.1], [0.0, 0.0]],  # Last position is padding
            ],
            device=device,
        )

        padding_mask = torch.tensor(
            [[False, False, False], [False, False, True]], device=device
        )

        result, batch_offsets = padded_to_concatenated(tensor, padding_mask)

        # Expected shape: [5, 2] (total non-padded elements, feature dim)
        assert result.shape == (5, 2)

        # Check values
        expected_values = torch.tensor(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1]], device=device
        )
        assert torch.allclose(result, expected_values)

        # Check batch offsets
        expected_offsets = torch.tensor([0, 3, 5], device=device)
        assert torch.equal(batch_offsets, expected_offsets)

    def test_without_padding_mask(self, device):
        """Test without a padding mask."""
        tensor = torch.tensor(
            [[[1.0, 1.1], [2.0, 2.1]], [[3.0, 3.1], [4.0, 4.1]]], device=device
        )

        result, batch_offsets = padded_to_concatenated(tensor)

        # All elements kept - shape [4, 2]
        assert result.shape == (4, 2)

        # Should be just a view when all sequences are equal length
        assert result[0, 0] == 1.0
        tensor[0, 0, 0] = 9.9
        assert result[0, 0] == 9.9

        # Check batch offsets
        expected_offsets = torch.tensor([0, 2, 4], device=device)
        assert torch.equal(batch_offsets, expected_offsets)

    def test_empty_batch(self, device):
        """Test with an empty batch."""
        # Create a batched tensor with padding
        tensor = torch.tensor(
            [
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[4.0, 4.1], [5.0, 5.1], [0.0, 0.0]],  # Last position is padding
            ],
            device=device,
        )

        padding_mask = torch.tensor(
            [[False, False, False], [True, True, True], [False, False, True]],
            device=device,
        )

        result, batch_offsets = padded_to_concatenated(tensor, padding_mask)

        # Expected shape: [5, 2] (total non-padded elements, feature dim)
        assert result.shape == (5, 2)

        # Check values
        expected_values = torch.tensor(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1]], device=device
        )
        assert torch.allclose(result, expected_values)

        # Check batch offsets
        expected_offsets = torch.tensor([0, 3, 3, 5], device=device)
        assert torch.equal(batch_offsets, expected_offsets)

    def test_scalar_feature(self, device):
        """Test with a 2D tensor of shape [batch, seq_len] and scalar feature"""
        # Create padded tensor
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]], device=device)
        pad_mask = torch.tensor(
            [[False, False, False], [False, True, True]], device=device
        )

        result, batch_offsets = padded_to_concatenated(tensor, pad_mask)

        assert result.shape == (4,)

        expected_result = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        expected_batch_offsets = torch.tensor([0, 3, 4], device=device)

        assert torch.equal(result, expected_result)
        assert torch.equal(batch_offsets, expected_batch_offsets)

    def test_error_handling(self, device):
        """Test error handling."""
        # Test with tensor with less than 3 dimensions
        tensor_1d = torch.rand(3, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected tensor to have at least 2 dimensions",
        ):
            padded_to_concatenated(tensor_1d)

        # Test with mismatched padding mask dimensions
        tensor = torch.rand(3, 4, 2, device=device)
        padding_mask_wrong_batch = torch.zeros(2, 4, device=device, dtype=torch.bool)
        padding_mask_wrong_batch[0, -1] = True
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Batch size mismatch"  # type: ignore
        ):
            padded_to_concatenated(tensor, padding_mask_wrong_batch)


@pytest.mark.cpu_and_cuda
class TestBatchDimToLeadingIndex:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device)

        result = batch_dim_to_leading_index(tensor)

        # Expected shape: [2, 2, 3] (original + 1 for batch index)
        assert result.shape == (2, 2, 3)

        # Check batch indices in the first position of the last dimension
        assert torch.equal(result[0, 0, 0], torch.tensor(0, device=device))
        assert torch.equal(result[0, 1, 0], torch.tensor(0, device=device))
        assert torch.equal(result[1, 0, 0], torch.tensor(1, device=device))
        assert torch.equal(result[1, 1, 0], torch.tensor(1, device=device))

        # Check original values in the remaining positions
        assert torch.equal(result[0, 0, 1:], torch.tensor([1, 2], device=device))
        assert torch.equal(result[0, 1, 1:], torch.tensor([3, 4], device=device))
        assert torch.equal(result[1, 0, 1:], torch.tensor([5, 6], device=device))
        assert torch.equal(result[1, 1, 1:], torch.tensor([7, 8], device=device))


@pytest.mark.cpu_and_cuda
class TestBatchOffsetsFromSparseTensorIndices:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        # Create indices for a sparse tensor with batch dimension first
        indices = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 3]],
            device=device,
        )

        result = batch_offsets_from_sparse_tensor_indices(indices)

        # Expected: [0, 2, 4, 6] - 3 batches of 2
        expected = torch.tensor([0, 2, 4, 6], device=device)
        assert torch.equal(result, expected)

    def test_non_contiguous_batches(self, device):
        """Test with non-contiguous batch indices."""
        # Sparse indices with gaps in batch indices (batch 1 missing)
        indices = torch.tensor(
            [
                [0, 0, 2, 2, 3],  # Batch indices: 0, 0, 2, 2, 3
                [0, 1, 0, 1, 0],  # Other indices
            ],
            device=device,
        )

        result = batch_offsets_from_sparse_tensor_indices(indices)

        # Each element is the first occurrence of that batch index
        expected = torch.tensor([0, 2, 2, 4, 5], device=device)
        assert torch.equal(result, expected)

    def test_empty_indices_tensor(self, device):
        indices = torch.zeros([0, 2], device=device, dtype=torch.long)

        expected = torch.zeros(1, device=device, dtype=torch.long)
        result = batch_offsets_from_sparse_tensor_indices(indices)

        assert torch.equal(expected, result)


@pytest.mark.cpu_and_cuda
class TestSparseTensorToPadded:
    def test_basic_functionality(self, device):
        """Test basic functions"""
        indices = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 2]], device=device).T
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        tensor = torch.sparse_coo_tensor(
            indices, values, (2, 3), device=device
        ).coalesce()

        padded_values, padded_indices, pad_mask = sparse_tensor_to_padded(tensor)

        assert padded_values.shape == (2, 2)
        assert padded_indices.shape == (2, 2, 2)
        assert pad_mask.shape == (2, 2)

        expected_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        expected_indices = indices.T.reshape(2, 2, 2)
        assert torch.equal(padded_values, expected_values)
        assert torch.equal(padded_indices, expected_indices)

        assert torch.equal(pad_mask, torch.zeros_like(pad_mask))

    def test_uneven_batches(self, device):
        """Test with uneven batch sizes anbd padding"""
        indices = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0]], device=device).T
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        tensor = torch.sparse_coo_tensor(
            indices, values, (2, 3), device=device
        ).coalesce()

        padded_values, padded_indices, pad_mask = sparse_tensor_to_padded(tensor)

        assert padded_values.shape == (2, 3)  # 2 batches, max 3 elements
        assert padded_indices.shape == (2, 3, 2)
        assert pad_mask.shape == (2, 3)

        assert torch.equal(
            padded_values[0], torch.tensor([1.0, 2.0, 3.0], device=device)
        )
        assert torch.equal(padded_values[1, 0], torch.tensor(4.0, device=device))

        expected_indices = torch.zeros_like(padded_indices)
        expected_indices[0] = expected_indices.new_tensor([[0, 0], [0, 1], [0, 2]])
        expected_indices[1, 0] = expected_indices.new_tensor([1, 0])
        assert torch.equal(padded_indices, expected_indices)

        expected_mask = torch.tensor(
            [[False, False, False], [False, True, True]], device=device
        )
        assert torch.equal(pad_mask, expected_mask)

    def test_error_not_sparse(self, device):
        """Test error when giving a non-sparse tensor"""
        tensor = torch.randn(2, 2, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Received non-sparse tensor",
        ):
            sparse_tensor_to_padded(tensor)


class TestPaddedToSparseTensor:
    def test_basic_functionality(self, device):
        """Test basic functions"""
        padded_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        padded_indices = torch.tensor(
            [[[0, 0], [0, 1]], [[1, 0], [1, 2]]], device=device
        )
        pad_mask = torch.zeros(2, 2, dtype=torch.bool, device=device)

        tensor = padded_to_sparse_tensor(
            padded_values, padded_indices, pad_mask, [2, 3]
        )

        assert tensor.is_sparse
        assert tensor.shape == (2, 3)

        expected_indices = padded_indices.reshape(-1, 2).T
        expected_values = padded_values.reshape(-1)

        assert torch.equal(tensor.indices(), expected_indices)
        assert torch.equal(tensor.values(), expected_values)

        # Test without shape param
        tensor_2 = padded_to_sparse_tensor(padded_values, padded_indices, pad_mask)

        # Should be same as other tensor
        assert tensor.shape == tensor_2.shape
        assert torch.equal(tensor.indices(), tensor_2.indices())
        assert torch.equal(tensor.values(), tensor_2.values())


@pytest.mark.cpu_and_cuda
class TestSparseTensorToConcatenated:
    def test_basic_functionality(self, device):
        """Test basic functions"""
        indices = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 2]], device=device).T
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        tensor = torch.sparse_coo_tensor(
            indices, values, (2, 3), device=device
        ).coalesce()

        concat_values, concat_indices, batch_offsets = sparse_tensor_to_concatenated(
            tensor
        )

        assert concat_values.shape == (4,)
        assert concat_indices.shape == (4, 2)

        expected_values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        expected_indices = indices.T
        assert torch.equal(concat_values, expected_values)
        assert torch.equal(concat_indices, expected_indices)

        assert torch.equal(batch_offsets, torch.tensor([0, 2, 4], device=device))

    def test_uneven_batches(self, device):
        """Test with uneven batch sizes anbd padding"""
        indices = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0]], device=device).T
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        tensor = torch.sparse_coo_tensor(
            indices, values, (2, 3), device=device
        ).coalesce()

        concat_values, concat_indices, batch_offsets = sparse_tensor_to_concatenated(
            tensor
        )

        assert concat_values.shape == (4,)
        assert concat_indices.shape == (4, 2)

        assert torch.equal(
            concat_values, torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        )

        expected_indices = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0]], device=device)
        assert torch.equal(concat_indices, expected_indices)

        assert torch.equal(batch_offsets, torch.tensor([0, 3, 4], device=device))

    def test_error_not_sparse(self, device):
        """Test error when giving a non-sparse tensor"""
        tensor = torch.randn(2, 2, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Received non-sparse tensor",
        ):
            sparse_tensor_to_concatenated(tensor)
