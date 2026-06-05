"""Publication-style CKA panels for VLA-Trace reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vla_trace.visualization.style import apply_publication_style, load_pyplot, save_figure_all


DATASETS = ("coco", "libero_10", "libero_goal", "libero_spatial", "libero_object")
DATASET_LABELS = {
    "coco": "COCO",
    "libero_10": "LIBERO-10",
    "libero_goal": "Goal",
    "libero_spatial": "Spatial",
    "libero_object": "Object",
}
MODEL_LABELS = {"pi05": r"$\pi_{0.5}$", "openvla": "OpenVLA", "openvla_oft": "OpenVLA-OFT"}
CKPTS = ("C0", "C1", "C2")
VIEWS = ("vision_pooled", "text_pooled", "joint_pooled")
VIEW_LABELS = {"vision_pooled": "Vision", "text_pooled": "Text", "joint_pooled": "Joint"}
CKPT_COLORS = {"C0": "#7F8C8D", "C1": "#1F77B4", "C2": "#D62728"}
CKPT_MARKERS = {"C0": "o", "C1": "s", "C2": "^"}
DRIFT_COLORS = {"C1_vs_C0": "#1F77B4", "C2_vs_C1": "#D62728", "C2_vs_C0": "#2CA02C"}
DRIFT_LABELS = {"C1_vs_C0": "C1 vs C0", "C2_vs_C1": "C2 vs C1", "C2_vs_C0": "C2 vs C0"}


def plot_cka_publication(
    reports: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    datasets: tuple[str, ...] = DATASETS,
    models: tuple[str, ...] = ("pi05", "openvla"),
) -> list[Path]:
    """Draw publication-style CKA panels from explicit report paths.

    `reports` uses keys such as `pi05:libero_10:alignment`,
    `openvla:libero_10:alignment`, and `openvla:libero_10:drift`.
    Missing reports are rendered as empty panels rather than inferred from local
    project paths.
    """
    if not reports:
        raise ValueError("plot_cka_publication requires at least one KEY=PATH report")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = {key: _read_json(path) for key, path in reports.items()}
    outputs: list[Path] = []
    plt = load_pyplot()
    apply_publication_style(plt)
    outputs.extend(_plot_image_text_panel(plt, payloads, output_root, datasets=datasets, models=models))
    for view in VIEWS:
        if _has_any_drift(payloads, view):
            outputs.extend(_plot_drift_panel(plt, payloads, output_root, datasets=datasets, models=models, view=view))
    if _has_any_drift(payloads, None):
        outputs.extend(_plot_drift_heatmap(plt, payloads, output_root, datasets=datasets, models=models))
        outputs.extend(_plot_publication_main(plt, payloads, output_root, datasets=datasets, models=models))
    return outputs


def _plot_image_text_panel(
    plt: Any,
    reports: dict[str, dict[str, Any]],
    output_root: Path,
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
) -> list[Path]:
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(max(4.0, 2.6 * len(datasets)), max(2.5, 2.55 * len(models))), sharey=True, squeeze=False, constrained_layout=True)
    for col, dataset in enumerate(datasets):
        axes[0, col].set_title(DATASET_LABELS.get(dataset, dataset), fontsize=10, fontweight="bold", pad=4)
    for row, model in enumerate(models):
        xticks, xlim = _model_xticks(model)
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            for ckpt in CKPTS:
                series = _image_text_series(reports.get(f"{model}:{dataset}:alignment"), ckpt)
                if series is None:
                    continue
                layers, mean, lo, hi = series
                color = CKPT_COLORS[ckpt]
                if not np.allclose(mean, lo) or not np.allclose(mean, hi):
                    ax.fill_between(layers, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(layers, mean, color=color, marker=CKPT_MARKERS[ckpt], markersize=3.0, label=ckpt)
            _format_cka_axis(ax, xticks, xlim)
            if row == len(models) - 1:
                ax.set_xlabel("Layer", labelpad=2)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS.get(model, model)}\nImage-Text CKA")
                if row == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(handles, labels, loc="upper right", frameon=False, ncol=3, handlelength=1.5, columnspacing=0.8)
    fig.suptitle("Layerwise Image-Text CKA Across Pretraining Stages and Datasets", fontsize=11.5, y=1.04)
    outputs = save_figure_all(fig, output_root / "image_text_cka_panel")
    plt.close(fig)
    return outputs


def _plot_drift_panel(
    plt: Any,
    reports: dict[str, dict[str, Any]],
    output_root: Path,
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    view: str,
) -> list[Path]:
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(max(4.0, 2.6 * len(datasets)), max(2.5, 2.55 * len(models))), sharey=True, squeeze=False, constrained_layout=True)
    for col, dataset in enumerate(datasets):
        axes[0, col].set_title(DATASET_LABELS.get(dataset, dataset), fontsize=10, fontweight="bold", pad=4)
    for row, model in enumerate(models):
        xticks, xlim = _model_xticks(model)
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            payload = reports.get(f"{model}:{dataset}:drift")
            for comparison in ("C1_vs_C0", "C2_vs_C1", "C2_vs_C0"):
                series = _drift_series(payload, view, comparison, dataset=dataset)
                if series is None:
                    continue
                layers, values = series
                ax.plot(layers, values, color=DRIFT_COLORS[comparison], linestyle="--" if comparison == "C2_vs_C0" else "-", marker="o", markersize=2.8, label=DRIFT_LABELS[comparison])
            _format_cka_axis(ax, xticks, xlim)
            if row == len(models) - 1:
                ax.set_xlabel("Layer", labelpad=2)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS.get(model, model)}\nDrift CKA")
                if row == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(handles, labels, loc="lower left", frameon=False, ncol=1, handlelength=1.5)
    fig.suptitle(f"Layerwise Representation Drift CKA - {VIEW_LABELS.get(view, view)} pooled view", fontsize=11.5, y=1.04)
    outputs = save_figure_all(fig, output_root / f"drift_cka_panel_{view}")
    plt.close(fig)
    return outputs


def _plot_drift_heatmap(
    plt: Any,
    reports: dict[str, dict[str, Any]],
    output_root: Path,
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
) -> list[Path]:
    fig, axes = plt.subplots(len(models), len(VIEWS), figsize=(max(7.0, 3.75 * len(VIEWS)), max(2.8, 2.45 * len(models))), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F2F2F2")
    last_im = None
    for row, model in enumerate(models):
        for col, view in enumerate(VIEWS):
            ax = axes[row, col]
            matrix = np.full((3, len(datasets)), np.nan, dtype=float)
            for c_idx, comparison in enumerate(("C1_vs_C0", "C2_vs_C1", "C2_vs_C0")):
                for d_idx, dataset in enumerate(datasets):
                    series = _drift_series(reports.get(f"{model}:{dataset}:drift"), view, comparison, dataset=dataset)
                    if series is not None:
                        matrix[c_idx, d_idx] = float(np.nanmean(series[1]))
            last_im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
            for r_idx in range(matrix.shape[0]):
                for c_idx in range(matrix.shape[1]):
                    value = matrix[r_idx, c_idx]
                    text = "n/a" if np.isnan(value) else f"{value:.2f}"
                    color = "#666666" if np.isnan(value) else ("white" if value > 0.6 else "#1A1A1A")
                    ax.text(c_idx, r_idx, text, ha="center", va="center", fontsize=7.5, color=color)
            ax.set_xticks(np.arange(len(datasets)), [DATASET_LABELS.get(d, d) for d in datasets], rotation=20, ha="right", fontsize=7.5)
            ax.set_yticks(np.arange(3), [DRIFT_LABELS[c] for c in ("C1_vs_C0", "C2_vs_C1", "C2_vs_C0")], fontsize=8)
            ax.set_xticks(np.arange(len(datasets) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(4) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(VIEW_LABELS[view], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(MODEL_LABELS.get(model, model), fontsize=10)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.7, pad=0.012, aspect=22)
        cbar.set_label("Mean drift CKA (layer-averaged)", fontsize=9)
    fig.suptitle("Representation Drift Summary: Layer-averaged CKA", fontsize=11.5, y=1.04)
    outputs = save_figure_all(fig, output_root / "drift_heatmap_summary")
    plt.close(fig)
    return outputs


def _plot_publication_main(
    plt: Any,
    reports: dict[str, dict[str, Any]],
    output_root: Path,
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
) -> list[Path]:
    """Draw the consolidated CKA main panel from explicit public reports."""
    fig = plt.figure(figsize=(max(7.0, 2.6 * len(datasets)), max(5.6, 2.35 + len(models) * 1.8)), constrained_layout=True)
    gs = fig.add_gridspec(len(models) + 1, len(datasets), height_ratios=[*[1.0] * len(models), 1.05])

    for row, model in enumerate(models):
        xticks, xlim = _model_xticks(model)
        for col, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[row, col])
            for ckpt in CKPTS:
                series = _image_text_series(reports.get(f"{model}:{dataset}:alignment"), ckpt)
                if series is None:
                    continue
                layers, mean, lo, hi = series
                color = CKPT_COLORS[ckpt]
                if not np.allclose(mean, lo) or not np.allclose(mean, hi):
                    ax.fill_between(layers, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(layers, mean, color=color, marker=CKPT_MARKERS[ckpt], markersize=2.8, label=ckpt)
            _format_cka_axis(ax, xticks, xlim)
            if row == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=10, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS.get(model, model)}\nImage-Text CKA")
            else:
                ax.set_yticklabels([])
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc="upper right", frameon=False, ncol=3, handlelength=1.4, columnspacing=0.7, fontsize=7.5)

    for col, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[len(models), col])
        for model in models:
            for comparison in ("C1_vs_C0", "C2_vs_C1", "C2_vs_C0"):
                series = _drift_series(reports.get(f"{model}:{dataset}:drift"), "vision_pooled", comparison, dataset=dataset)
                if series is None:
                    continue
                layers, values = series
                _, xlim = _model_xticks(model)
                norm_x = layers / max(xlim[1] - 0.5, 1.0)
                ax.plot(
                    norm_x,
                    values,
                    color=DRIFT_COLORS[comparison],
                    linestyle="-" if model == "pi05" else "--",
                    marker="o" if model == "pi05" else "s",
                    markersize=2.5,
                    linewidth=1.4,
                    alpha=0.9 if model == "pi05" else 0.7,
                )
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.4, alpha=0.5)
        ax.set_xlabel("Normalised depth", labelpad=2)
        if col == 0:
            ax.set_ylabel("Drift CKA\n(Vision)")
        else:
            ax.set_yticklabels([])

    legend_elems = [
        plt.Line2D([0], [0], color=DRIFT_COLORS["C1_vs_C0"], lw=1.6, label="C1 vs C0"),
        plt.Line2D([0], [0], color=DRIFT_COLORS["C2_vs_C1"], lw=1.6, label="C2 vs C1"),
        plt.Line2D([0], [0], color=DRIFT_COLORS["C2_vs_C0"], lw=1.6, linestyle="--", label="C2 vs C0"),
    ]
    if models:
        legend_elems.extend(
            [
                plt.Line2D([0], [0], color="#444444", lw=1.6, linestyle="-", marker="o", markersize=3, label=MODEL_LABELS.get(models[0], models[0])),
                *(
                    [
                        plt.Line2D([0], [0], color="#444444", lw=1.6, linestyle="--", marker="s", markersize=3, label=MODEL_LABELS.get(models[1], models[1])),
                    ]
                    if len(models) > 1
                    else []
                ),
            ]
        )
    fig.axes[-len(datasets)].legend(handles=legend_elems, loc="lower left", frameon=False, ncol=1, handlelength=1.6, fontsize=7.5)
    fig.suptitle("Image-Text Alignment and Representation Drift Across Pretraining Stages", fontsize=12, y=1.02)
    outputs = save_figure_all(fig, output_root / "cka_publication_main")
    plt.close(fig)
    return outputs


def _image_text_series(payload: dict[str, Any] | None, ckpt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not payload:
        return None
    if "metrics" in payload:
        baseline = payload.get("metrics", {}).get("cka", {}).get("baseline", {})
        mean_map = dict(baseline.get("mean", {}).get(ckpt, {}))
        if not mean_map:
            return None
        low_map = dict(baseline.get("ci_low", {}).get(ckpt, {}))
        high_map = dict(baseline.get("ci_high", {}).get(ckpt, {}))
    else:
        mean_map = dict(payload.get("profiles", {}).get(ckpt, {}))
        low_map = {}
        high_map = {}
    raw_layers = sorted(int(layer) for layer in mean_map)
    if raw_layers and raw_layers[0] == 0:
        layer_key_by_plot_layer = {layer + 1: str(layer) for layer in raw_layers}
    else:
        layer_key_by_plot_layer = {layer: str(layer) for layer in raw_layers}
    layers = sorted(layer_key_by_plot_layer)
    if not layers:
        return None
    mean = np.asarray([float(mean_map[layer_key_by_plot_layer[layer]]) for layer in layers], dtype=float)
    lo = np.asarray([float(low_map.get(layer_key_by_plot_layer[layer], mean_map[layer_key_by_plot_layer[layer]])) for layer in layers], dtype=float)
    hi = np.asarray([float(high_map.get(layer_key_by_plot_layer[layer], mean_map[layer_key_by_plot_layer[layer]])) for layer in layers], dtype=float)
    return np.asarray(layers, dtype=int), mean, lo, hi


def _drift_series(
    payload: dict[str, Any] | None,
    view: str,
    comparison: str,
    *,
    dataset: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not payload:
        return None
    summary_payload = _summary_drift_payload(payload, view, comparison, dataset=dataset)
    if summary_payload is not None:
        layers = summary_payload.get("anchor_layers") or summary_payload.get("layers")
        values = summary_payload.get("diag_cka")
        if layers is not None and values is not None:
            return np.asarray(layers, dtype=int), np.asarray(values, dtype=float)
    if "targets" in payload:
        target_key = {"C1_vs_C0": "C1", "C2_vs_C0": "C2", "C2_vs_C1": "C2_vs_C1"}.get(comparison)
        item = payload.get("targets", {}).get(target_key, {})
        view_payload = item.get("views", {}).get(view, {})
        layers = view_payload.get("anchor_layers") or view_payload.get("layers")
        values = view_payload.get("diag_cka")
        if layers is None or values is None:
            return None
        return np.asarray(layers, dtype=int), np.asarray(values, dtype=float)
    if "matched_layer_summary" in payload:
        if comparison == "C2_vs_C1":
            return None
        summary = payload["matched_layer_summary"]
        target = {"C1_vs_C0": "C1", "C2_vs_C0": "C2", "C2_vs_C1": "C2"}.get(comparison)
        view_payload = summary.get("targets", {}).get(target, {}).get(view, {})
        if view_payload.get("status", "ok") != "ok":
            return None
        per_layer = dict(view_payload.get("per_layer", {}))
        if not per_layer:
            return None
        layers = sorted(int(layer) for layer in per_layer)
        return np.asarray(layers, dtype=int), np.asarray([float(per_layer[str(layer)]) for layer in layers], dtype=float)
    return None


def _has_any_drift(reports: dict[str, dict[str, Any]], view: str | None) -> bool:
    for payload in reports.values():
        if view is None:
            if any(_drift_series(payload, candidate, "C1_vs_C0") is not None for candidate in VIEWS):
                return True
        elif _drift_series(payload, view, "C1_vs_C0") is not None:
            return True
    return False


def _summary_drift_payload(
    payload: dict[str, Any],
    view: str,
    comparison: str,
    *,
    dataset: str | None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    if dataset:
        for key in _dataset_summary_keys(dataset):
            item = payload.get(key)
            if isinstance(item, dict):
                candidates.append(item)
    candidates.append(payload)
    for candidate in candidates:
        view_payload = candidate.get(view)
        if isinstance(view_payload, dict) and isinstance(view_payload.get(comparison), dict):
            return view_payload[comparison]
    return None


def _dataset_summary_keys(dataset: str) -> tuple[str, ...]:
    return {
        "coco": ("COCO", "coco"),
        "libero_10": ("LIBERO-10", "libero_10", "libero"),
        "libero_goal": ("LIBERO-goal", "LIBERO-Goal", "libero_goal"),
        "libero_spatial": ("LIBERO-spatial", "LIBERO-Spatial", "libero_spatial"),
        "libero_object": ("LIBERO-object", "LIBERO-Object", "libero_object"),
    }.get(dataset, (dataset,))


def _format_cka_axis(ax: Any, xticks: tuple[int, ...], xlim: tuple[float, float]) -> None:
    ax.set_xticks(xticks)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.4, alpha=0.5)
    ax.axhspan(0.0, 0.2, color="#F5F5F5", zorder=-10)


def _model_xticks(model: str) -> tuple[tuple[int, ...], tuple[float, float]]:
    if model == "pi05":
        return (1, 6, 12, 18), (0.5, 18.5)
    return (1, 8, 16, 24, 32), (0.5, 32.5)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
