from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


TokenSelector = slice | Sequence[int]


@dataclass(frozen=True)
class TokenLayout:
    batch_axis: int = 0
    token_axis: int = 1
    feature_axis: int = -1
    groups: Mapping[str, TokenSelector] = field(default_factory=dict)


def _normalize_tokens(tokens: np.ndarray, layout: TokenLayout) -> np.ndarray:
    tokens = np.asarray(tokens)
    if tokens.ndim < 3:
        raise ValueError("tokens must have at least 3 dimensions")
    ndim = tokens.ndim
    axes = [layout.batch_axis % ndim, layout.token_axis % ndim, layout.feature_axis % ndim]
    if len(set(axes)) != 3:
        raise ValueError("layout axes must be distinct")
    return np.moveaxis(tokens, axes, (0, 1, 2))


def _pool_group(group_tokens: np.ndarray, mode: str) -> np.ndarray:
    if group_tokens.shape[1] == 0:
        raise ValueError("token group must not be empty")
    if mode == "mean":
        return group_tokens.mean(axis=1)
    if mode == "max":
        return group_tokens.max(axis=1)
    if mode == "first":
        return group_tokens[:, 0, :]
    raise ValueError(f"unsupported pooling mode: {mode}")


def pool_tokens(
    tokens: np.ndarray,
    layout: TokenLayout | None = None,
    *,
    mode: str = "mean",
    groups: Mapping[str, TokenSelector] | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    layout = layout or TokenLayout()
    normalized = _normalize_tokens(tokens, layout)
    selectors = dict(layout.groups)
    if groups is not None:
        selectors.update(groups)
    if not selectors:
        return _pool_group(normalized, mode=mode)
    pooled: dict[str, np.ndarray] = {}
    for name, selector in selectors.items():
        pooled[name] = _pool_group(normalized[:, selector, :], mode=mode)
    return pooled
