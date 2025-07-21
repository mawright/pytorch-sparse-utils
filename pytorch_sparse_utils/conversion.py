from typing import Optional, Union

import sparse
import torch
from torch import Tensor

from pytorch_sparse_utils import imports
from pytorch_sparse_utils.imports import ME, spconv


def torch_sparse_to_pydata_sparse(tensor: Tensor) -> sparse.COO:
    """Converts a sparse torch.Tensor to an equivalent Pydata sparse COO array

    Args:
        tensor (torch.Tensor): Sparse tensor to be converted

    Returns:
        array (sparse.COO): Pydata sparse COO array
    """
    assert tensor.is_sparse
    tensor = tensor.detach().cpu().coalesce()
    assert tensor.is_coalesced
    nonzero_values = tensor.values().nonzero(as_tuple=True)
    return sparse.COO(
        tensor.indices()[:, nonzero_values[0]].numpy(),
        tensor.values()[nonzero_values].numpy(),
        tensor.shape,
        has_duplicates=False,
    )


def pydata_sparse_to_torch_sparse(
    sparse_array: sparse.COO, device: Optional[Union[str, torch.device]] = None
) -> Tensor:
    """Converts a Pydata sparse COO array to an equivalent sparse torch.Tensor

    Args:
        sparse_array (sparse.COO): Pydata sparse COO array to be converted
        device (Optional[Union[str, torch.device]]): Device on which to create the
            sparse tensor. Defaults to None (default device).

    Returns:
        tensor (torch.Tensor): Converted sparse tensor
    """
    return torch.sparse_coo_tensor(
        indices=sparse_array.coords,  # pyright: ignore[reportArgumentType]
        values=sparse_array.data,  # pyright: ignore[reportArgumentType]
        size=sparse_array.shape,
        device=device,
    ).coalesce()


@imports.requires_minkowskiengine
def torch_sparse_to_minkowski(tensor: Tensor):
    """Converts a sparse torch.Tensor to an equivalent MinkowskiEngine SparseTensor

    Args:
        tensor (torch.Tensor): Sparse tensor to be converted

    Returns:
        sparse_tensor (MinkowskiEngine.SparseTensor): Converted sparse tensor
    """
    assert isinstance(tensor, Tensor)
    assert tensor.is_sparse
    features = tensor.values()
    if features.ndim == 1:
        features = features.unsqueeze(-1)
    coordinates = tensor.indices().T.int().contiguous()
    return ME.SparseTensor(
        features, coordinates, requires_grad=tensor.requires_grad, device=tensor.device
    )


@imports.requires_minkowskiengine
def minkowski_to_torch_sparse(
    tensor: Union[Tensor, ME.SparseTensor],
    full_scale_spatial_shape: Optional[Union[Tensor, list[int]]] = None,
    squeeze: bool = False
) -> Tensor:
    """Converts a MinkowskiEngine SparseTensor to an equivalent sparse torch.Tensor

    Args:
        tensor (MinkowskiEngine.SparseTensor): Sparse tensor to be converted
        full_scale_spatial_shape (Optional[Union[list[int], [Tensor]]]): The full
            extent of the spatial domain on which the sparse data reside.
            If given, will be used to define the size of the sparse tensor. If not
            given, the size will be inferred from the indices in the tensor.
            Default: None
        squeeze (bool): If True and the feature dimension of the MinkowskiEngine
            SparseTensor is 1, the returned sparse torch.Tensor will have its values
            squeezed to 1D shape of [nnz] rather than [nnz, 1]. Raises an error if
            True and the feature dimension is not 1.

    Returns:
        tensor (torch.Tensor): Converted sparse tensor
    """
    if isinstance(tensor, Tensor):
        assert tensor.is_sparse
        return tensor
    assert isinstance(tensor, ME.SparseTensor)
    min_coords = torch.zeros([tensor.dimension], dtype=torch.int, device=tensor.device)
    if full_scale_spatial_shape is not None:
        if isinstance(full_scale_spatial_shape, list):
            max_coords = torch.tensor(
                full_scale_spatial_shape, dtype=torch.int, device=tensor.device
            )
        else:
            assert isinstance(full_scale_spatial_shape, Tensor)
            max_coords = full_scale_spatial_shape.to(tensor.C)
    else:
        max_coords = None
    out = __me_sparse(tensor, min_coords, max_coords)[0].coalesce()
    if squeeze:
        if out.values().shape[1] != 1:
            raise ValueError(
                "Got `squeeze`=True, but the MinkowskiEngine tensor has a feature "
                f"dim of {out.values().shape[1]}, not 1."
            )
        out = torch.sparse_coo_tensor(
            out.indices(),
            out.values().squeeze(-1),
            out.shape[:-1],
            is_coalesced=out.is_coalesced()
        )
    return out


@imports.requires_spconv
def torch_sparse_to_spconv(tensor: torch.Tensor):
    """Converts a sparse torch.Tensor to an equivalent spconv SparseConvTensor

    Args:
        tensor (torch.Tensor): Sparse tensor to be converted

    Returns:
        SparseConvTensor (spconv.SparseConvTensor): Converted spconv tensor
    """
    if isinstance(tensor, spconv.SparseConvTensor):
        return tensor
    assert tensor.is_sparse
    spatial_shape = list(tensor.shape[1:-1])
    batch_size = tensor.shape[0]
    indices_th = tensor.indices()
    features_th = tensor.values()
    if features_th.ndim == 1:
        # Tensor has scalar features, but spconv always expects 2D feature tensor
        features_th = features_th.unsqueeze(-1)
        spatial_shape = spatial_shape + [tensor.shape[-1]]
    indices_th = indices_th.permute(1, 0).contiguous().int()
    return spconv.SparseConvTensor(features_th, indices_th, spatial_shape, batch_size)


@imports.requires_spconv
def spconv_to_torch_sparse(tensor, squeeze=False) -> Tensor:
    """Converts an spconv SparseConvTensor to a sparse torch.Tensor

    Args:
        tensor (spconv.SparseConvTensor): spconv tensor to be converted
        squeeze (bool): If the spconv tensor has a feature dimension of 1,
            setting this to true squeezes it out so that the resulting
            sparse Tensor has a dense_dim() of 0. Raises an error if the spconv
            feature dim is not 1.

    Returns:
        tensor (Tensor): Converted sparse torch.Tensor
    """
    if isinstance(tensor, Tensor) and tensor.is_sparse:
        return tensor
    assert isinstance(tensor, spconv.SparseConvTensor)
    if squeeze:
        if tensor.features.shape[-1] != 1:
            raise ValueError(
                "Got `squeeze`=True, but the spconv tensor has a feature dim of "
                f"{tensor.features.shape[-1]}, not 1"
            )
        size = [tensor.batch_size] + tensor.spatial_shape
        values = tensor.features.squeeze(-1)
    else:
        size = [tensor.batch_size] + tensor.spatial_shape + [tensor.features.shape[-1]]
        values = tensor.features
    indices = tensor.indices.transpose(0, 1)
    out = torch.sparse_coo_tensor(
        indices,
        values,
        size,
        device=tensor.features.device,
        dtype=tensor.features.dtype,
        requires_grad=tensor.features.requires_grad,
        check_invariants=True,
    )
    out = out.coalesce()
    return out


@imports.requires_minkowskiengine
def __me_sparse(
    tensor: ME.SparseTensor,
    min_coords: Optional[Tensor] = None,
    max_coords: Optional[Tensor] = None,
    contract_coords=True,
):
    r"""Copied from MinkowskiEngine's SparseTensor.sparse() method to fix
    device placement bugs.
    """
    if min_coords is not None:
        assert min_coords.dtype == torch.int
        assert min_coords.numel() == tensor._D
    if max_coords is not None:
        assert max_coords.dtype == torch.int
        assert max_coords.numel() == tensor._D

    def torch_sparse_Tensor(coords, feats, size=None):
        if size is None:
            if feats.dtype == torch.float64 or feats.dtype == torch.float32:
                return torch.sparse_coo_tensor(coords, feats, dtype=feats.dtype)
            else:
                raise ValueError("Feature type not supported.")
        else:
            if feats.dtype == torch.float64 or feats.dtype == torch.float32:
                return torch.sparse_coo_tensor(coords, feats, size, dtype=feats.dtype)
            else:
                raise ValueError("Feature type not supported.")

    # Use int tensor for all operations
    tensor_stride = torch.tensor(
        tensor.tensor_stride, dtype=torch.int, device=tensor.device
    )

    # New coordinates
    coords = tensor.C
    coords, batch_indices = coords[:, 1:], coords[:, 0]

    if min_coords is None:
        min_coords, _ = coords.min(0, keepdim=True)
    elif min_coords.ndim == 1:
        min_coords = min_coords.unsqueeze(0)

    assert (
        min_coords % tensor_stride
    ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

    if max_coords is not None:
        if max_coords.ndim == 1:
            max_coords = max_coords.unsqueeze(0)
        assert (
            max_coords % tensor_stride
        ).sum() == 0, "The maximum coordinates must be divisible by the tensor stride."

    coords -= min_coords

    if coords.ndim == 1:
        coords = coords.unsqueeze(1)
    if batch_indices.ndim == 1:
        batch_indices = batch_indices.unsqueeze(1)

    # return the contracted tensor
    if contract_coords:
        coords = coords // tensor_stride
        if max_coords is not None:
            max_coords = max_coords // tensor_stride
        min_coords = min_coords // tensor_stride

    new_coords = torch.cat((batch_indices, coords), dim=1).long()

    size = None
    if max_coords is not None:
        size = max_coords - min_coords
        # Squeeze to make the size one-dimensional
        size = size.squeeze()

        max_batch = tensor._manager.number_of_unique_batch_indices()
        size = torch.Size(
            [max_batch, *size, tensor.F.size(1)]  # pyright: ignore[reportArgumentType]
        )

    sparse_tensor = torch_sparse_Tensor(
        new_coords.t().to(tensor.F.device), tensor.F, size
    )
    tensor_stride = torch.tensor(
        tensor.tensor_stride, dtype=torch.int, device=tensor.device
    )
    return sparse_tensor, min_coords, tensor_stride
