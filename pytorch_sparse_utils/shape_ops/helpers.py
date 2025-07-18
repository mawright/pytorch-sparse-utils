import torch
from ..misc import prod, list_where

@torch.jit.script
def check_valid(shape: list[int], sparse_or_dense: str):
    dim_invalid = [x < -1 for x in shape]
    if any(dim_invalid):
        invalid_dims: list[int] = []
        for x, is_invalid in zip(shape, dim_invalid):
            if is_invalid:
                invalid_dims.append(x)
        raise RuntimeError(
            f"Invalid {sparse_or_dense} shape dimension(s) {invalid_dims}"
        )


@torch.jit.script
def do_infer_dim(
    new_shape: list[int],
    in_shape: list[int],
    sparse_or_dense: str,
) -> list[int]:
    to_infer: list[int] = list_where([x == -1 for x in new_shape])
    num_to_infer = len(to_infer)
    if num_to_infer == 0:
        return new_shape
    elif num_to_infer > 1:
        raise RuntimeError(
            f"Only one dimension can be inferred, but got new_{sparse_or_dense}_shape={new_shape}."
        )

    numel = prod(in_shape)
    if numel == 0:
        inferred_shape = new_shape.copy()
        inferred_shape[inferred_shape.index(-1)] = 0
        return inferred_shape

    partial_shape: list[int] = []
    for i in range(len(new_shape)):
        if i != to_infer[0]:
            partial_shape.append(new_shape[i])
    partial_new_numel = prod(partial_shape)
    if numel % partial_new_numel != 0:
        raise RuntimeError(
            f"New {sparse_or_dense} shape {new_shape} is invalid for input with {sparse_or_dense} shape {in_shape}."
        )

    inferred_shape = new_shape.copy()
    inferred_shape[to_infer[0]] = numel // partial_new_numel
    return inferred_shape


@torch.jit.script
def verify_shape_numel(
    new_shape: list[int], in_shape: list[int], sparse_or_dense: str
):
    in_numel = prod(in_shape)
    new_numel = prod(new_shape)
    if in_numel != new_numel:
        raise RuntimeError(
            f"New {sparse_or_dense} shape {new_shape} is invalid for input with {sparse_or_dense} shape {in_shape}."
        )
