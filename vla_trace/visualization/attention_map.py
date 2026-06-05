"""Generic attention-map plots for saved Stage 3 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vla_trace.visualization.style import apply_style, load_pyplot, save_figure


def plot_attention_map(
    array_path: str | Path,
    output: str | Path,
    *,
    key: str | None = None,
    keep_last_dims: int = 2,
    normalize: str = "none",
    kind: str = "auto",
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    title: str | None = None,
) -> Path:
    """Plot a saved 1D or 2D attention artifact as bars, lines, or a heatmap.

    Higher-rank tensors are averaged over leading dimensions. This supports
    common VLA-Trace diagnostics such as action-to-text bars, token-wise
    text-to-image matrices, and layer-wise modality summaries.
    """
    values = _reduce_attention(_load_array(array_path, key), keep_last_dims=keep_last_dims)
    values = _normalize(values, normalize)

    plt = load_pyplot()
    apply_style(plt)
    kind = _resolve_kind(values, kind)
    if values.ndim == 1 and kind == "bar":
        fig = _plot_bar(plt, values, x_labels=x_labels, title=title)
    elif values.ndim == 1 and kind == "line":
        fig = _plot_line(plt, values, x_labels=x_labels, y_labels=None, title=title)
    elif values.ndim == 2 and kind == "bar":
        fig = _plot_grouped_bar(plt, values, x_labels=x_labels, y_labels=y_labels, title=title)
    elif values.ndim == 2 and kind == "line":
        fig = _plot_line(plt, values, x_labels=x_labels, y_labels=y_labels, title=title)
    elif values.ndim == 2 and kind == "heatmap":
        fig = _plot_heatmap(plt, values, x_labels=x_labels, y_labels=y_labels, title=title)
    else:
        raise ValueError(f"Unsupported attention plot kind={kind!r} for shape {values.shape}")
    target = save_figure(fig, output)
    plt.close(fig)
    return target


def _load_array(path: str | Path, key: str | None) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False), dtype=np.float64)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            chosen = key or payload.files[0]
            return np.asarray(payload[chosen], dtype=np.float64)
    raise ValueError(f"Unsupported attention artifact format: {source.suffix}")


def _reduce_attention(array: np.ndarray, *, keep_last_dims: int) -> np.ndarray:
    if keep_last_dims not in {1, 2}:
        raise ValueError("keep_last_dims must be 1 or 2")
    if array.ndim <= keep_last_dims:
        return array
    axes = tuple(range(array.ndim - keep_last_dims))
    return np.nanmean(array, axis=axes)


def _normalize(values: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    array = np.asarray(values, dtype=np.float64)
    if mode == "none":
        return array
    if mode == "sum":
        total = float(np.nansum(array))
        return array / total if total > 1e-12 else array
    if mode == "max":
        max_value = float(np.nanmax(array)) if array.size else 0.0
        return array / max_value if max_value > 1e-12 else array
    if mode == "row":
        if array.ndim != 2:
            raise ValueError("row normalization requires a 2D attention map")
        denom = np.nansum(array, axis=1, keepdims=True)
        return np.divide(array, denom, out=np.zeros_like(array), where=denom > 1e-12)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _resolve_kind(values: np.ndarray, kind: str) -> str:
    kind = kind.lower()
    if kind == "auto":
        return "bar" if values.ndim == 1 else "heatmap"
    if kind not in {"bar", "line", "heatmap"}:
        raise ValueError(f"Unsupported attention plot kind: {kind}")
    if values.ndim == 1 and kind == "heatmap":
        raise ValueError("heatmap kind requires a 2D attention map")
    return kind


def _plot_bar(plt: Any, values: np.ndarray, *, x_labels: list[str] | None, title: str | None) -> Any:
    fig, ax = plt.subplots(figsize=(max(4.8, values.size * 0.36), 3.0))
    x_values = np.arange(values.size)
    ax.bar(x_values, values, color="#3B75AF", width=0.82)
    ax.set_title(title or "Attention")
    ax.set_ylabel("Attention")
    ax.set_xticks(x_values, _fit_labels(x_labels, values.size), rotation=45, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    return fig


def _plot_grouped_bar(
    plt: Any,
    values: np.ndarray,
    *,
    x_labels: list[str] | None,
    y_labels: list[str] | None,
    title: str | None,
) -> Any:
    series, width = values.shape
    fig, ax = plt.subplots(figsize=(max(5.0, width * 0.42), 3.2))
    x_values = np.arange(width)
    labels = _fit_labels(y_labels, series)
    bar_width = min(0.82 / max(series, 1), 0.28)
    offsets = (np.arange(series) - (series - 1) / 2.0) * bar_width
    for idx in range(series):
        ax.bar(x_values + offsets[idx], values[idx], width=bar_width, label=labels[idx])
    ax.set_title(title or "Attention")
    ax.set_ylabel("Attention")
    ax.set_xticks(x_values, _fit_labels(x_labels, width), rotation=45, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_line(
    plt: Any,
    values: np.ndarray,
    *,
    x_labels: list[str] | None,
    y_labels: list[str] | None,
    title: str | None,
) -> Any:
    array = values.reshape(1, -1) if values.ndim == 1 else values
    series, width = array.shape
    fig, ax = plt.subplots(figsize=(max(5.0, width * 0.36), 3.2))
    x_values = np.arange(width)
    labels = _fit_labels(y_labels, series) if values.ndim == 2 else ["attention"]
    for idx in range(series):
        ax.plot(x_values, array[idx], marker="o", linewidth=1.3, markersize=3.0, label=labels[idx])
    ax.set_title(title or "Attention")
    ax.set_ylabel("Attention")
    ax.set_xticks(x_values, _fit_labels(x_labels, width), rotation=45, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    if values.ndim == 2:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_heatmap(
    plt: Any,
    values: np.ndarray,
    *,
    x_labels: list[str] | None,
    y_labels: list[str] | None,
    title: str | None,
) -> Any:
    height, width = values.shape
    fig, ax = plt.subplots(figsize=(max(4.5, width * 0.36), max(3.2, height * 0.28)))
    image = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_title(title or "Attention Map")
    ax.set_xticks(np.arange(width), _fit_labels(x_labels, width), rotation=45, ha="right")
    ax.set_yticks(np.arange(height), _fit_labels(y_labels, height))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="Attention")
    fig.tight_layout()
    return fig


def _fit_labels(labels: list[str] | None, size: int) -> list[str]:
    if labels is None:
        return [str(index) for index in range(size)]
    if len(labels) != size:
        raise ValueError(f"Expected {size} labels, got {len(labels)}")
    return labels
