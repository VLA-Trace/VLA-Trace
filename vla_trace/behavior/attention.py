"""Attention localization metrics for Stage 3 behavior traces."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_GRID_SHAPE = (16, 16)
METRIC_FIELDS = [
    "trace_id",
    "model",
    "dataset",
    "task_id",
    "phase",
    "step_id",
    "instance_name",
    "category",
    "attn_mass",
    "iou_top10_gt",
    "iou_top10",
    "iou_fixed_gt",
    "iou_fixedthr",
    "peak_hit",
    "center_dist",
    "attn_entropy",
    "mask_patch_count",
    "valid_attention",
]


def attention_to_grid(attention: Any, grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE) -> np.ndarray:
    """Convert a vector or high-rank attention block into a 2D patch grid.

    Accepted inputs include `(patches,)`, `(H, W)`, or tensors whose last
    dimension is `H*W`; all leading dimensions are averaged.
    """
    array = np.asarray(attention, dtype=np.float64)
    patches = grid_shape[0] * grid_shape[1]
    if array.ndim == 1:
        if array.size != patches:
            side = int(math.sqrt(array.size))
            if side * side != array.size:
                raise ValueError(f"Cannot reshape attention vector of length {array.size} into a square grid")
            grid_shape = (side, side)
        return array.reshape(grid_shape)
    if array.ndim == 2 and array.shape == grid_shape:
        return array
    if array.shape[-1] == patches:
        return array.reshape(-1, patches).mean(axis=0).reshape(grid_shape)
    if array.shape[-2:] == grid_shape:
        return array.reshape(-1, *grid_shape).mean(axis=0)
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return array
    raise ValueError(f"Unsupported attention shape {array.shape}; expected vector or patch grid")


def mask_to_grid(mask: Any, grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE) -> np.ndarray:
    """Convert a bool mask to the attention patch grid using nearest sampling."""
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 1:
        if array.size != grid_shape[0] * grid_shape[1]:
            raise ValueError(f"Cannot reshape mask vector of length {array.size} into {grid_shape}")
        return array.reshape(grid_shape).astype(bool)
    if array.ndim != 2:
        raise ValueError(f"Mask must be 1D or 2D, got shape {array.shape}")
    if array.shape == grid_shape:
        return array.astype(bool)
    row_idx = np.linspace(0, array.shape[0] - 1, grid_shape[0]).round().astype(int)
    col_idx = np.linspace(0, array.shape[1] - 1, grid_shape[1]).round().astype(int)
    return array[np.ix_(row_idx, col_idx)].astype(bool)


def attention_mass(attention: np.ndarray, mask: np.ndarray, *, eps: float = 1e-12) -> float:
    total = float(np.nansum(attention))
    if total <= eps:
        return float("nan")
    if not mask.any():
        return 0.0
    return float(np.nansum(attention[mask]) / total)


def iou_top_percent(attention: np.ndarray, mask: np.ndarray, *, top_percent: float = 10.0) -> float:
    if not mask.any():
        return float("nan")
    percentile = 100.0 - float(top_percent)
    threshold = float(np.nanpercentile(attention, percentile))
    attention_mask = attention >= threshold
    union = attention_mask | mask
    if not union.any():
        return float("nan")
    return float((attention_mask & mask).sum() / union.sum())


def iou_fixed_threshold(attention: np.ndarray, mask: np.ndarray, *, frac: float = 0.5) -> float:
    if not mask.any():
        return float("nan")
    max_value = float(np.nanmax(attention)) if attention.size else 0.0
    if max_value <= 1e-12:
        return float("nan")
    attention_mask = attention >= max_value * float(frac)
    union = attention_mask | mask
    if not union.any():
        return float("nan")
    return float((attention_mask & mask).sum() / union.sum())


def peak_hit(attention: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any() or attention.size == 0:
        return 0.0
    index = np.unravel_index(int(np.nanargmax(attention)), attention.shape)
    return float(bool(mask[index]))


def attention_entropy(attention: np.ndarray, *, eps: float = 1e-12) -> float:
    values = np.asarray(attention, dtype=np.float64).reshape(-1)
    total = float(np.nansum(values))
    if total <= eps:
        return float("nan")
    probs = values / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def center_distance(attention: np.ndarray, mask: np.ndarray, *, eps: float = 1e-12) -> float:
    if not mask.any():
        return float("nan")
    total = float(np.nansum(attention))
    rows, cols = np.mgrid[: attention.shape[0], : attention.shape[1]]
    if total <= eps:
        attn_center = (attention.shape[0] / 2.0, attention.shape[1] / 2.0)
    else:
        attn_center = (
            float(np.nansum(rows * attention) / total),
            float(np.nansum(cols * attention) / total),
        )
    mask_rows, mask_cols = np.where(mask)
    mask_center = (float(mask_rows.mean()), float(mask_cols.mean()))
    return float(math.sqrt((attn_center[0] - mask_center[0]) ** 2 + (attn_center[1] - mask_center[1]) ** 2))


def compute_attention_metrics(
    attention: Any,
    masks: Mapping[str, Any],
    *,
    categories: Mapping[str, str] | None = None,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    top_percent: float = 10.0,
    fixed_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    attention_grid = attention_to_grid(attention, grid_shape)
    valid_attention = bool(np.nansum(attention_grid) > 1e-12)
    entropy = attention_entropy(attention_grid)
    categories = categories or {}
    rows = []
    for name, mask in masks.items():
        mask_grid = mask_to_grid(mask, grid_shape)
        rows.append(
            {
                "instance_name": str(name),
                "category": categories.get(str(name), "unknown"),
                "attn_mass": attention_mass(attention_grid, mask_grid),
                "iou_top10": iou_top_percent(attention_grid, mask_grid, top_percent=top_percent),
                "iou_fixedthr": iou_fixed_threshold(attention_grid, mask_grid, frac=fixed_threshold),
                "peak_hit": peak_hit(attention_grid, mask_grid),
                "center_dist": center_distance(attention_grid, mask_grid),
                "attn_entropy": entropy,
                "mask_patch_count": int(mask_grid.sum()),
                "valid_attention": valid_attention,
            }
        )
        rows[-1]["iou_top10_gt"] = rows[-1]["iou_top10"]
        rows[-1]["iou_fixed_gt"] = rows[-1]["iou_fixedthr"]
    return rows


def compute_attention_metrics_from_files(
    attention_path: str | Path,
    mask_path: str | Path,
    output_csv: str | Path,
    *,
    metadata_path: str | Path | None = None,
    objects_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    top_percent: float = 10.0,
    fixed_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    attention_maps = _load_npz(attention_path)
    mask_maps = _load_npz(mask_path)
    metadata = _load_json(metadata_path) if metadata_path else {}
    categories_by_step = _load_step_categories(objects_path) if objects_path else {}
    masks_by_step: dict[int, dict[str, np.ndarray]] = {}
    for key, value in mask_maps.items():
        step, instance = parse_step_mask_key(key)
        masks_by_step.setdefault(step, {})[instance] = value

    all_rows: list[dict[str, Any]] = []
    for step in sorted(masks_by_step):
        attention_key = step_key(step)
        if attention_key not in attention_maps:
            continue
        phase = phase_for_step(step, metadata)
        rows = compute_attention_metrics(
            attention_maps[attention_key],
            masks_by_step[step],
            categories=categories_by_step.get(step, {}),
            grid_shape=grid_shape,
            top_percent=top_percent,
            fixed_threshold=fixed_threshold,
        )
        for row in rows:
            row.update(
                {
                    "trace_id": metadata.get("trace_id", ""),
                    "model": metadata.get("model") or metadata.get("model_type", ""),
                    "dataset": metadata.get("dataset", ""),
                    "task_id": metadata.get("task_id", ""),
                    "phase": phase,
                    "step_id": step,
                }
            )
        all_rows.extend(rows)

    write_metric_csv(output_csv, all_rows)
    if summary_path:
        _write_json(summary_path, summarize_attention_rows(all_rows))
    return all_rows


def summarize_attention_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("phase", "full")), str(row.get("category", "unknown"))), []).append(row)
    summaries = []
    for (phase, category), group in sorted(groups.items()):
        summaries.append(
            {
                "phase": phase,
                "category": category,
                "n_rows": len(group),
                "mean_attn_mass": _nanmean(row["attn_mass"] for row in group),
                "mean_iou_top10": _nanmean(row["iou_top10"] for row in group),
                "mean_peak_hit": _nanmean(row["peak_hit"] for row in group),
                "invalid_attention": sum(1 for row in group if not row.get("valid_attention")),
                "empty_masks": sum(1 for row in group if int(row.get("mask_patch_count", 0)) == 0),
            }
        )
    return {"n_rows": len(rows), "groups": summaries}


def write_metric_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys([*METRIC_FIELDS, *(key for row in rows for key in row)]))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_plot_summary_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = ("iou_top10_gt", "iou_fixed_gt", "attn_mass"),
) -> Path:
    """Write the long-form CSV consumed by `vla-trace plot-attention`."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        phase = str(row.get("phase", "full"))
        for metric in metrics:
            value = _metric_value(row, metric)
            if value is None:
                continue
            value_f = float(value)
            if not np.isfinite(value_f):
                continue
            key = (task_id, phase, metric)
            groups.setdefault(key, []).append(value_f)
            meta.setdefault(
                key,
                {
                    "trace_id": row.get("trace_id", ""),
                    "model": row.get("model", ""),
                    "dataset": row.get("dataset", ""),
                },
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trace_id",
        "model",
        "dataset",
        "task_id",
        "phase",
        "phase_label",
        "metric",
        "metric_label",
        "mean_iou",
        "std_iou",
        "n_steps",
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, values in sorted(groups.items()):
            task_id, phase, metric = key
            array = np.asarray(values, dtype=np.float64)
            writer.writerow(
                {
                    **meta.get(key, {}),
                    "task_id": task_id,
                    "phase": phase,
                    "phase_label": phase.replace("_", " ").title(),
                    "metric": metric,
                    "metric_label": _metric_label(metric),
                    "mean_iou": f"{float(array.mean()):.6f}",
                    "std_iou": f"{float(array.std(ddof=0)):.6f}",
                    "n_steps": int(array.size),
                }
            )
    return target


def step_key(step: int) -> str:
    return f"step_{int(step):03d}"


def parse_step_mask_key(key: str) -> tuple[int, str]:
    match = re.match(r"^step_(?P<step>\d+)_(?P<name>.+)$", str(key))
    if not match:
        raise ValueError(f"Expected mask key like step_030_object, got {key!r}")
    return int(match.group("step")), match.group("name")


def phase_for_step(step: int, metadata: Mapping[str, Any]) -> str:
    phases = metadata.get("phases")
    if isinstance(phases, list):
        for item in phases:
            start = int(item.get("start_step", 0))
            stop = int(item.get("stop_step", item.get("end_step", 10**12)))
            if start <= step < stop:
                return str(item.get("name", item.get("phase", "phase")))
    split = metadata.get("split_step")
    if split is not None:
        return "phase1" if step < int(split) else "phase2"
    total = metadata.get("total_steps")
    if total and int(total) > 1:
        return "phase1" if step < int(total) / 2 else "phase2"
    return "full"


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_step_categories(path: str | Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, dict[str, str]] = {}
    if isinstance(payload, list):
        for item in payload:
            step = int(item.get("step", item.get("step_id", 0)))
            mapping: dict[str, str] = {}
            for obj in item.get("objects", []):
                name = str(obj.get("mask_key_suffix") or obj.get("instance_name") or obj.get("name"))
                mapping[name] = str(obj.get("category", "object"))
            out[step] = mapping
    elif isinstance(payload, dict):
        for step_text, values in payload.items():
            step = int(str(step_text).replace("step_", ""))
            out[step] = {str(name): str(category) for name, category in dict(values).items()}
    return out


def _write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")
    return target


def _nanmean(values: Any) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(array.mean())


def _metric_label(metric: str) -> str:
    labels = {
        "iou_top10_gt": "Top-10 patch IoU",
        "iou_top10": "Top-10 IoU",
        "iou_fixed_gt": "Fixed-threshold IoU",
        "iou_fixedthr": "Fixed-threshold IoU",
        "attn_mass": "Attention Mass",
        "peak_hit": "Peak Hit",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def _metric_value(row: Mapping[str, Any], metric: str) -> Any:
    aliases = {
        "iou_top10_gt": "iou_top10",
        "iou_fixed_gt": "iou_fixedthr",
        "iou_top10": "iou_top10_gt",
        "iou_fixedthr": "iou_fixed_gt",
    }
    if metric in row:
        return row.get(metric)
    alias = aliases.get(metric)
    return row.get(alias) if alias else None
