import torch
import pytest

from pytorch_sparse_utils.conversion import (
    torch_sparse_to_minkowski,
    torch_sparse_to_spconv,
)
from pytorch_sparse_utils.imports import has_minkowskiengine, ME, has_spconv, spconv

from pytorch_sparse_utils.minkowskiengine import (
    MinkowskiGELU,
    MinkowskiLayerNorm,
    get_me_layer,
    MinkowskiNonlinearityBase,
)
from pytorch_sparse_utils.spconv import spconv_sparse_mult


@pytest.mark.skipif(not has_minkowskiengine, reason="MinkowskiEngine not installed")
@pytest.mark.cpu_and_cuda
class TestMinkowskiEngineUtils:
    def test_minkowski_layer_norm(self, device):
        indices = torch.tensor([[0, 0], [0, 1]], device=device).T
        values = torch.randn(2, 8, device=device)
        tensor = torch.sparse_coo_tensor(indices, values).coalesce()

        me_tensor = torch_sparse_to_minkowski(tensor)

        norm = MinkowskiLayerNorm(8).to(device)

        out = norm(me_tensor)
        assert isinstance(out, ME.SparseTensor)
        assert not torch.equal(me_tensor.F, out.F)
        assert torch.equal(me_tensor.C, out.C)

        me_tensor_field = ME.TensorField(me_tensor.F, me_tensor.C)

        out_2 = norm(me_tensor_field)
        assert isinstance(out_2, ME.TensorField)
        assert not torch.equal(me_tensor_field.F, out_2.F)
        assert torch.equal(me_tensor_field.C, out_2.C)

    def test_minkowski_gelu(self, device):
        indices = torch.tensor([[0, 0], [0, 1]], device=device).T
        values = torch.randn(2, 8, device=device)
        tensor = torch.sparse_coo_tensor(indices, values).coalesce()

        me_tensor = torch_sparse_to_minkowski(tensor)

        gelu = MinkowskiGELU()
        assert isinstance(gelu, MinkowskiNonlinearityBase)

        out = gelu(me_tensor)

        assert isinstance(out, ME.SparseTensor)
        assert not torch.equal(me_tensor.F, out.F)
        assert torch.equal(me_tensor.C, out.C)

    def test_get_me_layer(self):
        module = get_me_layer(MinkowskiGELU())  # pyright: ignore[reportArgumentType]
        assert isinstance(module, MinkowskiGELU)

        relu = get_me_layer("relu")
        assert isinstance(relu(), ME.MinkowskiReLU)

        gelu = get_me_layer("gelu")
        assert isinstance(gelu(), MinkowskiGELU)

        bn = get_me_layer("batchnorm1d")
        assert isinstance(
            bn(8), ME.MinkowskiBatchNorm  # pyright: ignore[reportCallIssue]
        )

        with pytest.raises(ValueError, match="Unexpected layer"):
            get_me_layer("fdsfdsf")


@pytest.mark.skipif(not has_spconv, reason="spconv not installed")
@pytest.mark.cpu_and_cuda
class TestSpConvUtils:
    def test_spconv_sparse_mult(self, device):
        indices = torch.tensor([[0, 0], [0, 1]], device=device).T
        values = torch.randn(2, 8, device=device)
        tensor = torch.sparse_coo_tensor(indices, values).coalesce()

        spconv_tensor = torch_sparse_to_spconv(tensor)
        assert isinstance(spconv_tensor, spconv.SparseConvTensor)

        out = spconv_sparse_mult(spconv_tensor, spconv_tensor)

        assert torch.equal(
            out.indices, spconv_tensor.indices  # pyright: ignore[reportArgumentType]
        )
        assert not torch.equal(out.features, spconv_tensor.features)

    def test_spconv_sparse_mult_different_indices(self, device):
        indices = torch.tensor([[0, 0], [1, 1], [1, 2]], device=device).T
        values = torch.randn(3, 8, device=device)
        tensor = torch.sparse_coo_tensor(indices, values, (2, 5, 8)).coalesce()

        spconv_tensor_1 = torch_sparse_to_spconv(tensor)

        indices = torch.tensor([[1, 0], [0, 1], [0, 0]], device=device).T
        values = torch.randn(3, 8, device=device)
        tensor = torch.sparse_coo_tensor(indices, values, (2, 5, 8)).coalesce()

        spconv_tensor_2 = torch_sparse_to_spconv(tensor)

        assert not torch.equal(
            spconv_tensor_1.indices,  # pyright: ignore[reportArgumentType]
            spconv_tensor_2.indices,  # pyright: ignore[reportArgumentType]
        )

        out = spconv_sparse_mult(spconv_tensor_1, spconv_tensor_2)

        assert not torch.equal(
            out.features,
            spconv_tensor_1.features,  # pyright: ignore[reportArgumentType]
        )
        assert not torch.equal(
            out.features,
            spconv_tensor_2.features,  # pyright: ignore[reportArgumentType]
        )
