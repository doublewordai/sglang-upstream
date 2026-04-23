from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def _to_flat_list(values: Any) -> List[int]:
    if values is None:
        return []
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        if values and isinstance(values[0], (list, tuple)):
            flattened: List[int] = []
            for row in values:
                flattened.extend(int(x) for x in row)
            return flattened
        return [int(x) for x in values]
    return [int(values)]


def _to_square_mask(values: Any) -> List[List[bool]]:
    if values is None:
        return []
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        mask = values.detach().cpu().numpy()
    elif isinstance(values, np.ndarray):
        mask = values
    elif hasattr(values, "tolist"):
        mask = np.asarray(values.tolist())
    else:
        mask = np.asarray(values)
    if mask.ndim != 2:
        return []
    return mask.astype(bool).tolist()


def build_ssd_verify_payload(
    *,
    current_token: int,
    prefix_len: int,
    num_draft_tokens: int,
    draft_result: Optional[Any],
) -> dict[str, List[int] | List[bool]]:
    draft_tokens = _to_flat_list(
        getattr(draft_result, "draft_tokens", None) if draft_result is not None else None
    )
    positions = _to_flat_list(
        getattr(draft_result, "positions", None) if draft_result is not None else None
    )
    retrieve_index = _to_flat_list(
        getattr(draft_result, "retrieve_index", None)
        if draft_result is not None
        else None
    )
    retrieve_next_token = _to_flat_list(
        getattr(draft_result, "retrieve_next_token", None)
        if draft_result is not None
        else None
    )
    retrieve_next_sibling = _to_flat_list(
        getattr(draft_result, "retrieve_next_sibling", None)
        if draft_result is not None
        else None
    )
    tree_mask = _to_square_mask(
        getattr(draft_result, "tree_mask", None) if draft_result is not None else None
    )

    exact_remote_tree = (
        draft_result is not None
        and int(getattr(draft_result, "num_tokens", 0)) == num_draft_tokens
        and len(draft_tokens) == num_draft_tokens
        and len(positions) == num_draft_tokens
        and len(retrieve_index) == num_draft_tokens
        and len(retrieve_next_token) == num_draft_tokens
        and len(retrieve_next_sibling) == num_draft_tokens
        and len(tree_mask) == num_draft_tokens
        and all(len(row) == num_draft_tokens for row in tree_mask)
    )

    if exact_remote_tree:
        token_list = draft_tokens
        position_list = positions
        retrieve_index_list = retrieve_index
        retrieve_next_token_list = retrieve_next_token
        retrieve_next_sibling_list = retrieve_next_sibling
        ancestor_mask = tree_mask
    else:
        token_list = [current_token] * num_draft_tokens
        position_list = [prefix_len + i for i in range(num_draft_tokens)]
        retrieve_index_list = list(range(num_draft_tokens))
        retrieve_next_token_list = [
            i + 1 if i + 1 < num_draft_tokens else -1 for i in range(num_draft_tokens)
        ]
        retrieve_next_sibling_list = [-1] * num_draft_tokens
        ancestor_mask = [
            [col <= row for col in range(num_draft_tokens)]
            for row in range(num_draft_tokens)
        ]

    full_mask: List[bool] = []
    for row in ancestor_mask:
        full_mask.extend([True] * prefix_len)
        full_mask.extend(bool(x) for x in row)

    return {
        "draft_tokens": token_list,
        "positions": position_list,
        "retrieve_index": retrieve_index_list,
        "retrieve_next_token": retrieve_next_token_list,
        "retrieve_next_sibling": retrieve_next_sibling_list,
        "custom_mask": full_mask,
    }
