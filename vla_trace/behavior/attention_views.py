"""Route-level attention extraction from raw model attention tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def extract_attention_views(
    attention: Any,
    *,
    visual_span: slice,
    text_span: slice,
    action_span: slice,
    normalize_rows: bool = True,
) -> dict[str, np.ndarray]:
    """Extract qualitative attention views from raw attention tensors.

    Accepted shapes are `[layers, heads, query, key]`, `[heads, query, key]`,
    or `[batch, layers, heads, query, key]`. The returned arrays are directly
    plottable by `plot-attention-map` or `attention-overlay`.
    """
    attn = _normalize_attention_shape(attention)
    if normalize_rows:
        denom = attn.sum(axis=-1, keepdims=True)
        attn = np.divide(attn, denom, out=np.zeros_like(attn), where=denom > 1e-12)

    q_action = _select(attn, action_span, axis=-2)
    action_to_image = q_action[..., visual_span].mean(axis=(0, 1, 2))
    action_to_text = q_action[..., text_span].mean(axis=(0, 1, 2))

    q_text = _select(attn, text_span, axis=-2)
    text_to_image = q_text[..., visual_span].mean(axis=(0, 1))

    layer_modality_mass = _layer_modality_mass(
        attn,
        query_span=action_span,
        key_spans={
            "image": visual_span,
            "text": text_span,
            "action": action_span,
        },
    )
    layer_modality_flow = _layer_modality_flow(
        attn,
        query_spans={
            "image": visual_span,
            "text": text_span,
            "action": action_span,
        },
        key_spans={
            "image": visual_span,
            "text": text_span,
            "action": action_span,
        },
    )
    return {
        "action_to_image": action_to_image.astype(np.float32),
        "action_to_text": action_to_text.astype(np.float32),
        "text_to_image": text_to_image.astype(np.float32),
        "layer_modality_mass": layer_modality_mass.astype(np.float32),
        "layer_modality_flow": layer_modality_flow.astype(np.float32),
    }


def write_attention_view_artifact(
    attention_path: str | Path,
    output_path: str | Path,
    *,
    key: str | None,
    visual_span: slice,
    text_span: slice,
    action_span: slice,
    normalize_rows: bool = True,
) -> Path:
    attention = _load_array(attention_path, key)
    views = extract_attention_views(
        attention,
        visual_span=visual_span,
        text_span=text_span,
        action_span=action_span,
        normalize_rows=normalize_rows,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **views)
    return target


def parse_span(value: str) -> slice:
    if ":" not in value:
        raise ValueError(f"Expected START:STOP span, got {value!r}")
    start_text, stop_text = value.split(":", 1)
    start = int(start_text) if start_text else None
    stop = int(stop_text) if stop_text else None
    return slice(start, stop)


def _normalize_attention_shape(attention: Any) -> np.ndarray:
    array = np.asarray(attention, dtype=np.float64)
    if array.ndim == 5:
        array = array.mean(axis=0)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected attention with 3, 4, or 5 dims, got shape {array.shape}")
    return array


def _select(array: np.ndarray, span: slice, *, axis: int) -> np.ndarray:
    index = [slice(None)] * array.ndim
    index[axis] = span
    return array[tuple(index)]


def _layer_modality_mass(attn: np.ndarray, *, query_span: slice, key_spans: dict[str, slice]) -> np.ndarray:
    q = _select(attn, query_span, axis=-2)
    rows = []
    for layer in range(q.shape[0]):
        layer_values = []
        for span in key_spans.values():
            layer_values.append(float(q[layer, ..., span].sum(axis=-1).mean()))
        rows.append(layer_values)
    return np.asarray(rows, dtype=np.float64)


def _layer_modality_flow(attn: np.ndarray, *, query_spans: dict[str, slice], key_spans: dict[str, slice]) -> np.ndarray:
    matrices = []
    for layer in range(attn.shape[0]):
        matrix = []
        for query_span in query_spans.values():
            q = _select(attn[layer : layer + 1], query_span, axis=-2)
            row = []
            for key_span in key_spans.values():
                row.append(float(q[..., key_span].sum(axis=-1).mean()))
            matrix.append(row)
        matrices.append(matrix)
    return np.asarray(matrices, dtype=np.float64)


def _load_array(path: str | Path, key: str | None) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            chosen = key or payload.files[0]
            return np.asarray(payload[chosen])
    raise ValueError(f"Unsupported attention artifact format: {source.suffix}")
