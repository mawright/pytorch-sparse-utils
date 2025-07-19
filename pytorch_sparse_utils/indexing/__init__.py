from .basics import batch_sparse_index, sparse_index_select, sparse_select
from .misc import union_sparse_indices

from .scatter import scatter_to_sparse_tensor

from .unique import unique_rows

from .utils import (
    flatten_nd_indices,
    flatten_sparse_indices,
    gather_mask_and_fill,
    get_sparse_index_mapping,
    linearize_sparse_and_index_tensors,
    unflatten_nd_indices,
)

__all__ = [
    # Basic operations
    "sparse_select",
    "sparse_index_select",
    # Bulk indexing
    "batch_sparse_index",
    # Sparse tensor scatter
    "scatter_to_sparse_tensor",
    # Miscellaneous functions
    "unique_rows",
    "union_sparse_indices",
    # Indexing helpers
    "flatten_nd_indices",
    "unflatten_nd_indices",
    "flatten_sparse_indices",
    "linearize_sparse_and_index_tensors",
    "get_sparse_index_mapping",
    "gather_mask_and_fill",
]
