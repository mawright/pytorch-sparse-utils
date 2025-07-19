import torch
from torch import Tensor


@torch.jit.script
def __sparse_or_dense(tensor: Tensor) -> str:
    return "sparse" if tensor.is_sparse else "dense"


@torch.jit.script
def union_sparse_indices(
    sparse_tensor_1: Tensor, sparse_tensor_2: Tensor
) -> tuple[Tensor, Tensor]:
    """Creates unified sparse tensors with the union of indices from both input tensors.

    This function takes two sparse tensors and returns versions of them that share the
    same set of indices (the union of indices from both inputs). For indices present in
    only one of the tensors, zeros are filled in for the corresponding values in the
    other tensor.

    This function is useful for ensuring a one-to-one correspondence between two
    sparse tensors' respective values() tensors, which in turn may be useful for
    elementwise value comparisons like loss functions.

    Args:
        sparse_tensor_1 (Tensor): First sparse tensor.
        sparse_tensor_2 (Tensor): Second sparse tensor with the same sparse and dense
            dimensions as sparse_tensor_1.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - tensor_1_unioned (Tensor): First tensor with indices expanded to include
                all indices from the second tensor (with zeros for missing values).
            - tensor_2_unioned (Tensor): Second tensor with indices expanded to include
                all indices from the first tensor (with zeros for missing values).

    Raises:
        ValueError: If either input is not a sparse tensor or if the sparse and
            dense dimensions don't match between tensors.

    Note:
        For very large sparse tensors, this operation may require significant memory
        for intermediate tensors.
    """
    if not sparse_tensor_1.is_sparse or not sparse_tensor_2.is_sparse:
        raise ValueError(
            "Expected two sparse tensors; got "
            f"{__sparse_or_dense(sparse_tensor_1)} and {__sparse_or_dense(sparse_tensor_2)}"
        )
    if sparse_tensor_1.shape != sparse_tensor_2.shape:
        raise ValueError(
            "Expected tensors to have same shapes; got "
            f"{sparse_tensor_1.shape} and {sparse_tensor_2.shape}"
        )
    if sparse_tensor_1.sparse_dim() != sparse_tensor_2.sparse_dim():
        raise ValueError(
            "Expected both sparse tensors to have equal numbers of sparse dims; got "
            f"{sparse_tensor_1.sparse_dim()} and {sparse_tensor_2.sparse_dim()}"
        )
    if sparse_tensor_1.dense_dim() != sparse_tensor_2.dense_dim():
        raise ValueError(
            "Expected both sparse tensors to have equal numbers of dense dims; got "
            f"{sparse_tensor_1.dense_dim()} and {sparse_tensor_2.dense_dim()}"
        )

    M = sparse_tensor_1.sparse_dim()
    K = sparse_tensor_1.dense_dim()

    sparse_tensor_1 = sparse_tensor_1.coalesce()
    sparse_tensor_2 = sparse_tensor_2.coalesce()

    indices_1, values_1 = sparse_tensor_1.indices(), sparse_tensor_1.values()
    indices_2, values_2 = sparse_tensor_2.indices(), sparse_tensor_2.values()

    # Need to find all indices that are unique to each sparse tensor
    # To do this, stack one of them twice and the other once
    indices_2_2_1 = torch.cat([indices_2, indices_2, indices_1], -1)
    uniques, counts = torch.unique(indices_2_2_1, dim=-1, return_counts=True)
    # Any that appear twice in the stacked indices are unique to tensor 2
    # and any that appear once are unique to tensor 1
    # (indices that appear 3x are shared already)
    indices_only_in_tensor_1 = uniques[:, counts == 1]
    indices_only_in_tensor_2 = uniques[:, counts == 2]

    # Figure out how many new indices will be added to each sparse tensor
    n_exclusives_1 = indices_only_in_tensor_1.size(-1)
    n_exclusives_2 = indices_only_in_tensor_2.size(-1)

    # Make zero-padding for new values tensors
    pad_zeros_1 = values_1.new_zeros(
        (n_exclusives_2,) + sparse_tensor_1.shape[M : M + K]
    )
    pad_zeros_2 = values_2.new_zeros(
        (n_exclusives_1,) + sparse_tensor_1.shape[M : M + K]
    )

    # Make the new tensors by stacking indices and values together
    tensor_1_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_1, indices_only_in_tensor_2], -1),
        torch.cat([values_1, pad_zeros_1], 0),
        size=sparse_tensor_1.shape,
        device=sparse_tensor_1.device,
    ).coalesce()

    tensor_2_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_2, indices_only_in_tensor_1], -1),
        torch.cat([values_2, pad_zeros_2], 0),
        size=sparse_tensor_2.shape,
        device=sparse_tensor_2.device,
    ).coalesce()

    if not torch.equal(tensor_1_unioned.indices(), tensor_2_unioned.indices()):
        raise RuntimeError("Internal error: unioned tensors have different indices")

    return tensor_1_unioned, tensor_2_unioned
