"""Stage 3 behavior visualizations derived from public artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vla_trace.behavior.attention import attention_to_grid, mask_to_grid
from vla_trace.visualization.style import apply_style, load_pyplot, save_figure


def plot_attention_overlay(
    image_path: str | Path,
    attention_path: str | Path,
    mask_path: str | Path | None,
    output: str | Path,
    *,
    attention_key: str | None = None,
    mask_key: str | None = None,
    grid_shape: tuple[int, int] = (16, 16),
    top_percent: float = 10.0,
    alpha: float = 0.55,
) -> Path:
    image = _load_array(image_path, key=None)
    attention = _load_array(attention_path, key=attention_key)
    masks = _load_optional_masks(mask_path, mask_key)
    heatmap = attention_to_grid(attention, grid_shape=grid_shape)

    plt = load_pyplot()
    apply_style(plt)
    fig, axes = plt.subplots(1, 2 if masks else 1, figsize=(8.2 if masks else 4.4, 4.0), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(_normalize_image(image))
    upsampled = _nearest_resize(heatmap, image.shape[:2])
    threshold = np.nanpercentile(heatmap, 100.0 - float(top_percent))
    alpha_mask = _nearest_resize((heatmap >= threshold).astype(float), image.shape[:2])
    im = ax.imshow(upsampled, cmap="jet", alpha=alpha_mask * float(alpha), interpolation="nearest")
    ax.set_title(f"Attention Top {top_percent:g}%")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    if masks:
        mask_ax = axes[0, 1]
        mask_rgb = _mask_panel(masks, image.shape[:2], grid_shape=grid_shape)
        mask_ax.imshow(mask_rgb, interpolation="nearest")
        mask_ax.set_title("Masks")
        mask_ax.axis("off")

    fig.tight_layout()
    target = save_figure(fig, output)
    plt.close(fig)
    return target


def _load_array(path: str | Path, key: str | None) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            chosen = key or payload.files[0]
            return np.asarray(payload[chosen])
    raise ValueError(f"Unsupported array artifact format: {source.suffix}")


def _load_optional_masks(path: str | Path | None, key: str | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    source = Path(path)
    if source.suffix == ".npy":
        return {key or "mask": np.asarray(np.load(source, allow_pickle=False))}
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if key:
                return {key: np.asarray(payload[key])}
            return {name: np.asarray(payload[name]) for name in payload.files}
    raise ValueError(f"Unsupported mask artifact format: {source.suffix}")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    max_value = float(np.nanmax(array)) if array.size else 1.0
    min_value = float(np.nanmin(array)) if array.size else 0.0
    if max_value <= 1.0 and min_value >= 0.0:
        return np.clip(array, 0.0, 1.0)
    return np.clip((array - min_value) / max(max_value - min_value, 1e-12), 0.0, 1.0)


def _nearest_resize(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(int)
    return array[np.ix_(rows, cols)]


def _mask_panel(
    masks: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    *,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    palette = np.asarray(
        [
            [0.85, 0.20, 0.18],
            [0.20, 0.55, 0.85],
            [0.20, 0.70, 0.35],
            [0.75, 0.45, 0.85],
            [0.95, 0.65, 0.20],
        ]
    )
    canvas = np.zeros((*image_shape, 3), dtype=np.float64)
    for idx, mask in enumerate(masks.values()):
        grid = mask_to_grid(mask, grid_shape=grid_shape)
        up = _nearest_resize(grid.astype(float), image_shape) > 0
        canvas[up] = palette[idx % len(palette)]
    return canvas
