import functools

import torch
from packaging import version

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiNonlinearityBase

    has_minkowskiengine = True
except ImportError:  # pragma: no cover
    class DummyME:
        SparseTensor = None
    ME = DummyME()
    MinkowskiNonlinearityBase = None
    has_minkowskiengine = False
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor

    has_spconv = True
except ImportError:  # pragma: no cover
    spconv = None
    SparseConvTensor = None
    has_spconv = False


def requires_minkowskiengine(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not has_minkowskiengine:
            raise ModuleNotFoundError("Could not import MinkowskiEngine")
        return func(*args, **kwargs)

    return wrapper


def requires_spconv(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not has_spconv:
            raise ModuleNotFoundError("Could not import spconv")
        return func(*args, **kwargs)

    return wrapper


def check_pytorch_version(required_version: str) -> bool:
    try:
        return version.parse(torch.__version__) >= version.parse(required_version)
    except ValueError as e:
        raise ValueError(f"Invalid version string: {required_version}") from e
