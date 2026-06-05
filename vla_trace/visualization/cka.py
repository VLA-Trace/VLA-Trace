"""Plot CKA reports produced by the public Stage 1 tools."""

from __future__ import annotations

import csv
import json
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from vla_trace.visualization.style import annotate_heatmap, apply_style, load_pyplot, save_figure


def plot_cka_report(
    report_path: str | Path,
    output: str | Path,
    *,
    title: str | None = None,
    source_data: str | Path | None = None,
) -> Path:
    """Draw a public CKA figure from a VLA-Trace report JSON."""
    payload = _read_json(report_path)
    _validate_report(payload, report_path)
    rows = cka_source_rows(payload, report_path)
    if source_data:
        _write_csv(source_data, rows)

    plt = load_pyplot()
    apply_style(plt)
    if _is_layerwise(payload):
        fig = _plot_layerwise(plt, payload, title)
    elif payload.get("analysis") == "checkpoint_drift":
        fig = _plot_similarity_heatmap(plt, payload, title)
    else:
        fig = _plot_cross_modal_heatmaps(plt, payload, title)

    target = save_figure(fig, output)
    plt.close(fig)
    return target


def cka_source_rows(payload: dict[str, Any], report_path: str | Path = "") -> list[dict[str, Any]]:
    """Return normalized CKA rows useful for source-data CSVs and tests."""
    analysis = str(payload.get("analysis", "cka"))
    report = str(report_path)
    rows: list[dict[str, Any]] = []
    if _is_layerwise(payload):
        for checkpoint, profile in dict(payload.get("profiles", {})).items():
            for layer, value in _sorted_layer_items(profile):
                rows.append(
                    {
                        "analysis": analysis,
                        "checkpoint": checkpoint,
                        "layer": layer,
                        "cka": float(value),
                        "view": payload.get("view", ""),
                        "source_json": report,
                    }
                )
        return rows

    if analysis == "checkpoint_drift" and "similarity" in payload:
        similarity = dict(payload["similarity"])
        for left, right_map in similarity.items():
            for right, value in dict(right_map).items():
                rows.append(
                    {
                        "analysis": analysis,
                        "left": left,
                        "right": right,
                        "cka": float(value),
                        "drift": 1.0 - float(value),
                        "view": payload.get("view", ""),
                        "source_json": report,
                    }
                )
        return rows

    for checkpoint, matrix in dict(payload.get("profiles", {})).items():
        for left, right_map in dict(matrix).items():
            for right, value in dict(right_map).items():
                rows.append(
                    {
                        "analysis": analysis,
                        "checkpoint": checkpoint,
                        "left_view": left,
                        "right_view": right,
                        "cka": float(value),
                        "source_json": report,
                    }
                )
    return rows


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_report(payload: dict[str, Any], report_path: str | Path) -> None:
    status = payload.get("status", "ok")
    if status != "ok":
        message = payload.get("message", f"CKA report is not ready: {report_path}")
        raise ValueError(str(message))
    if payload.get("analysis") == "checkpoint_drift":
        if not (payload.get("similarity") or payload.get("profiles")):
            raise ValueError("checkpoint-drift report must contain `similarity` or layerwise `profiles`")
        return
    if not payload.get("profiles"):
        raise ValueError("cross-modal report must contain `profiles`")


def _is_layerwise(payload: dict[str, Any]) -> bool:
    return bool(payload.get("profiles")) and (bool(payload.get("layerwise")) or _has_flat_layer_profiles(payload))


def _has_flat_layer_profiles(payload: dict[str, Any]) -> bool:
    profiles = dict(payload.get("profiles", {}))
    if not profiles:
        return False
    for profile in profiles.values():
        if not isinstance(profile, dict) or not profile:
            return False
        if not all(_is_number(value) for value in profile.values()):
            return False
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _sorted_layer_items(profile: dict[str, Any]) -> list[tuple[int, float]]:
    return sorted((int(layer), float(value)) for layer, value in dict(profile).items())


def _plot_layerwise(plt: Any, payload: dict[str, Any], title: str | None) -> Any:
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for checkpoint, profile in dict(payload.get("profiles", {})).items():
        items = _sorted_layer_items(profile)
        if not items:
            continue
        layers = [layer for layer, _ in items]
        values = [value for _, value in items]
        ax.plot(layers, values, marker="o", linewidth=1.3, markersize=3.0, label=str(checkpoint))

    ax.set_title(title or _default_title(payload))
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_similarity_heatmap(plt: Any, payload: dict[str, Any], title: str | None) -> Any:
    labels, matrix = _matrix_from_nested_dict(dict(payload["similarity"]))
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title or _default_title(payload))
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Target checkpoint")
    ax.set_ylabel("Reference checkpoint")
    annotate_heatmap(ax, matrix)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="CKA")
    fig.tight_layout()
    return fig


def _plot_cross_modal_heatmaps(plt: Any, payload: dict[str, Any], title: str | None) -> Any:
    profiles = dict(payload["profiles"])
    count = len(profiles)
    fig_width = max(3.3, 2.8 * count)
    fig, axes = plt.subplots(1, count, figsize=(fig_width, 3.1), squeeze=False, constrained_layout=True)
    axes_flat = list(axes.reshape(-1))
    last_image = None
    for ax, (checkpoint, profile) in zip(axes_flat, profiles.items()):
        labels, matrix = _matrix_from_nested_dict(dict(profile))
        last_image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_title(str(checkpoint))
        ax.set_xticks(np.arange(len(labels)), labels=_short_labels(labels), rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels=_short_labels(labels))
        annotate_heatmap(ax, matrix)

    fig.suptitle(title or _default_title(payload), y=1.02)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes_flat, fraction=0.025, pad=0.04, label="CKA")
    return fig


def _matrix_from_nested_dict(profile: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    labels = list(profile)
    right_labels = []
    for value in profile.values():
        for key in dict(value):
            if key not in right_labels:
                right_labels.append(key)
    if labels == right_labels:
        all_labels = labels
    else:
        all_labels = list(dict.fromkeys([*labels, *right_labels]))
    matrix = np.full((len(all_labels), len(all_labels)), np.nan, dtype=np.float64)
    for row, left in enumerate(all_labels):
        right_map = dict(profile.get(left, {}))
        for col, right in enumerate(all_labels):
            if right in right_map:
                matrix[row, col] = float(right_map[right])
    return all_labels, np.nan_to_num(matrix, nan=0.0)


def _short_labels(labels: list[str]) -> list[str]:
    return [
        label.replace("_pooled", "").replace("_", " ")
        for label in labels
    ]


def _default_title(payload: dict[str, Any]) -> str:
    analysis = str(payload.get("analysis", "cka")).replace("_", " ").title()
    if payload.get("view"):
        return f"{analysis} ({payload['view']})"
    return analysis


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No CKA rows to write")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
