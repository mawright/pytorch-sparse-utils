from typing import Optional, Union

import torch
from torch import Tensor
import math
from hypothesis import strategies as st

@st.composite
def random_sparse_tensor_strategy(draw: st.DrawFn, max_dim: Optional[int] = None):
    sparse_shape = draw(st.lists(st.integers(0, max_dim), min_size=1, max_size=4))
    dense_shape = draw(st.lists(st.integers(0, max_dim), min_size=0, max_size=3))
    sparsity = draw(st.floats(0.0, 1.0))
    seed = draw(st.integers(0, 2 ** 32 - 1))
    dtype = draw(st.sampled_from([torch.float32, torch.float16, torch.int32, torch.int64]))

    return {
        "sparse_shape": sparse_shape,
        "dense_shape": dense_shape,
        "sparsity": sparsity,
        "seed": seed,
        "dtype": dtype
    }


def random_sparse_tensor(
    sparse_shape: Union[list[int], tuple[int, ...]],
    dense_shape: Union[list[int], tuple[int, ...]],
    sparsity: float,
    seed: int,
    max_nnz: int = 1000,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    assert 0.0 <= sparsity <= 1.0
    if device is None:
        device = torch.get_default_device()
    device = torch.device(device)

    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    nnz = min(
        int(sparsity * math.prod(sparse_shape)),
        max_nnz
    )

    indices = (
        torch.rand((len(sparse_shape), nnz), device=device)
        * torch.tensor(sparse_shape, device=device).unsqueeze(-1)
    ).long()
    values = torch.empty([nnz] + list(dense_shape), device=device, dtype=dtype)
    if torch.is_floating_point(values):
        values.normal_()
    else:
        values.random_()
    tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size = list(sparse_shape) + list(dense_shape),
        requires_grad=requires_grad,
    ).coalesce()

    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state, device)
    else:
        torch.set_rng_state(rng_state)

    return tensor
