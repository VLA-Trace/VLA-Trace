from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _flatten_samples(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.asarray(array, dtype=np.float64)


def linear_cka(x: Any, y: Any, *, eps: float = 1e-12) -> float:
    x_array = _flatten_samples(x)
    y_array = _flatten_samples(y)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must share the same sample axis")
    x_centered = x_array - x_array.mean(axis=0, keepdims=True)
    y_centered = y_array - y_array.mean(axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    self_x = x_centered.T @ x_centered
    self_y = y_centered.T @ y_centered
    hsic_xy = float(np.sum(cross * cross))
    hsic_xx = float(np.sum(self_x * self_x))
    hsic_yy = float(np.sum(self_y * self_y))
    denominator = max(np.sqrt(hsic_xx * hsic_yy), eps)
    return float(hsic_xy / denominator)


def cross_modal_cka_profile(
    representations: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    names = list(representations)
    profile: dict[str, dict[str, float]] = {name: {} for name in names}
    for left_name in names:
        for right_name in names:
            profile[left_name][right_name] = linear_cka(
                representations[left_name],
                representations[right_name],
            )
    return profile


def layerwise_cross_modal_cka(
    representations: Mapping[str, Any],
    *,
    left_view: str = "vision_pooled",
    right_view: str = "text_pooled",
) -> dict[int, float]:
    """Compute CKA(left_view, right_view) for matched layers.

    The input may be nested as `{view: {layer: matrix}}` or flattened as
    `{f"{view}/layer_{layer}": matrix}`.
    """
    left = _extract_layer_map(representations, left_view)
    right = _extract_layer_map(representations, right_view)
    layers = sorted(set(left) & set(right))
    if not layers:
        raise ValueError(f"No matched layers for {left_view!r} and {right_view!r}")
    return {layer: float(linear_cka(left[layer], right[layer])) for layer in layers}


def layerwise_checkpoint_drift_cka(
    anchor: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    view: str = "joint_pooled",
) -> dict[int, float]:
    """Compute matched-layer checkpoint-drift CKA for one pooled view."""
    left = _extract_layer_map(anchor, view)
    right = _extract_layer_map(target, view)
    layers = sorted(set(left) & set(right))
    if not layers:
        raise ValueError(f"No matched layers for view {view!r}")
    return {layer: float(linear_cka(left[layer], right[layer])) for layer in layers}


def matched_layer_checkpoint_drift_summary(
    banks: Mapping[str, Mapping[str, Any]],
    *,
    reference: str | None = None,
    views: Iterable[str] = ("vision_pooled", "text_pooled", "joint_pooled"),
) -> dict[str, Any]:
    """Summarize matched-layer CKA drift across pooled views.

    This mirrors the publication-style checkpoint drift table: each target checkpoint
    is compared with a reference checkpoint at matched layer indices, then the
    per-layer CKA scores are averaged for each modality view.
    """
    if not banks:
        raise ValueError("banks must not be empty")
    views = tuple(views)
    names = list(banks)
    reference_name = reference or names[0]
    if reference_name not in banks:
        raise KeyError(f"unknown reference checkpoint: {reference_name}")

    summary: dict[str, Any] = {
        "reference": reference_name,
        "views": list(views),
        "targets": {},
    }
    anchor = banks[reference_name]
    for target_name, target in banks.items():
        target_views: dict[str, Any] = {}
        for view in views:
            try:
                scores = layerwise_checkpoint_drift_cka(anchor, target, view=view)
            except ValueError as exc:
                target_views[view] = {
                    "status": "missing",
                    "message": str(exc),
                    "n_layers": 0,
                    "layers": [],
                }
                continue
            values = np.asarray(list(scores.values()), dtype=np.float64)
            mean_cka = float(np.nanmean(values)) if values.size else float("nan")
            target_views[view] = {
                "status": "ok",
                "mean_cka": mean_cka,
                "mean_drift": 1.0 - mean_cka,
                "n_layers": int(values.size),
                "layers": list(scores),
                "per_layer": scores,
            }
        summary["targets"][target_name] = target_views
    return summary


def checkpoint_drift_cka(
    checkpoints: Mapping[str, Any],
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    if not checkpoints:
        raise ValueError("checkpoints must not be empty")
    names = list(checkpoints)
    reference_name = reference or names[0]
    if reference_name not in checkpoints:
        raise KeyError(f"unknown reference checkpoint: {reference_name}")

    similarity = cross_modal_cka_profile(checkpoints)
    drift = {
        name: 1.0 - similarity[reference_name][name]
        for name in names
    }
    consecutive = []
    for left_name, right_name in zip(names, names[1:]):
        cka_value = similarity[left_name][right_name]
        consecutive.append(
            {
                "from": left_name,
                "to": right_name,
                "cka": cka_value,
                "drift": 1.0 - cka_value,
            }
        )
    return {
        "reference": reference_name,
        "similarity": similarity,
        "drift": drift,
        "consecutive": consecutive,
    }


def _extract_layer_map(representations: Mapping[str, Any], view: str) -> dict[int, Any]:
    if view in representations and isinstance(representations[view], Mapping):
        return {int(layer): matrix for layer, matrix in representations[view].items()}
    prefix = f"{view}/layer_"
    out: dict[int, Any] = {}
    for name, matrix in representations.items():
        if str(name).startswith(prefix):
            out[int(str(name)[len(prefix):])] = matrix
    return out
