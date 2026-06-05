"""Plot rollout attention summaries from public trace artifacts."""

from __future__ import annotations

import csv
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vla_trace.visualization.style import apply_publication_style, load_pyplot, save_figure


@dataclass(frozen=True)
class AttentionIouRow:
    task_id: str
    phase: str
    phase_label: str
    metric: str
    metric_label: str
    mean_iou: float
    std_iou: float
    n_steps: int
    instruction: str = ""
    success: str = ""
    total_steps: int | None = None


def plot_attention_iou(
    csv_path: str | Path,
    output: str | Path,
    *,
    metric: str | None = None,
    title: str | None = None,
) -> Path:
    """Draw phase-wise attention IoU curves from a normalized CSV artifact."""
    rows = read_attention_iou_csv(csv_path)
    if metric:
        rows = [row for row in rows if _metric_matches(row.metric, metric)]
    if not rows:
        raise ValueError("No attention IoU rows found. Expected task_id, phase, metric, and mean_iou columns.")

    plt = load_pyplot()
    apply_publication_style(plt)
    fig = _plot_rows(plt, rows, title=title)
    target = save_figure(fig, output)
    plt.close(fig)
    return target


def read_attention_iou_csv(csv_path: str | Path) -> list[AttentionIouRow]:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames or [])
        rows = list(reader)
    return [_parse_row(row) for row in rows]


def _validate_columns(fieldnames: list[str]) -> None:
    columns = set(fieldnames)
    task_ok = bool({"task_id", "task"} & columns)
    phase_ok = bool({"phase", "step_group"} & columns)
    value_ok = bool({"mean_iou", "iou", "value"} & columns)
    missing = []
    if not task_ok:
        missing.append("task_id")
    if not phase_ok:
        missing.append("phase")
    if "metric" not in columns:
        missing.append("metric")
    if not value_ok:
        missing.append("mean_iou")
    if missing:
        raise ValueError(f"Attention IoU CSV is missing required column(s): {', '.join(missing)}")


def _parse_row(row: dict[str, str]) -> AttentionIouRow:
    task_id = row.get("task_id") or row.get("task") or "task"
    phase = row.get("phase") or row.get("step_group") or "phase"
    metric = row.get("metric") or "iou"
    return AttentionIouRow(
        task_id=str(task_id),
        phase=str(phase),
        phase_label=row.get("phase_label") or str(phase).replace("_", " "),
        metric=str(metric),
        metric_label=row.get("metric_label") or _metric_label(str(metric)),
        mean_iou=float(row.get("mean_iou") or row.get("iou") or row.get("value") or 0.0),
        std_iou=float(row.get("std_iou") or row.get("std") or 0.0),
        n_steps=int(float(row.get("n_steps") or row.get("count") or 0)),
        instruction=row.get("instruction") or "",
        success=row.get("success") or "",
        total_steps=_optional_int(row.get("total_steps")),
    )


def _plot_rows(plt: Any, rows: list[AttentionIouRow], *, title: str | None) -> Any:
    task_ids = sorted({row.task_id for row in rows}, key=_natural_task_key)
    metrics = _metric_order(rows)
    cols = min(5, max(1, len(task_ids)))
    rows_n = (len(task_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(max(4.0, cols * 2.04), max(2.7, rows_n * 2.55)), squeeze=False, sharey=True)
    axes_flat = list(axes.reshape(-1))
    y_max = max((row.mean_iou for row in rows if not math.isnan(row.mean_iou)), default=0.3)
    y_upper = min(1.0, max(0.36, y_max * 1.22))

    for ax, task_id in zip(axes_flat, task_ids):
        task_rows = [row for row in rows if row.task_id == task_id]
        phases = _phase_order(task_rows)
        x_values = list(range(len(phases)))
        for idx, metric in enumerate(metrics):
            metric_rows = {
                row.phase: row
                for row in task_rows
                if row.metric == metric
            }
            values = [metric_rows[phase].mean_iou if phase in metric_rows else float("nan") for phase in phases]
            labels = [metric_rows[phase].metric_label for phase in phases if phase in metric_rows]
            label = labels[0] if labels else metric.replace("_", " ")
            style = _metric_style(metric, idx)
            ax.plot(
                x_values,
                values,
                label=label,
                **style,
            )
            if metric == metrics[0]:
                for x_value, y_value in zip(x_values, values):
                    if not math.isnan(y_value):
                        ax.annotate(
                            f"{y_value:.2f}",
                            (x_value, y_value),
                            textcoords="offset points",
                            xytext=(0, 4),
                            ha="center",
                            va="bottom",
                            color=style.get("color", "black"),
                            fontsize=5.7,
                        )
        ax.set_title(_task_title(task_id, task_rows), pad=3.0)
        ax.set_xticks(x_values, [_phase_label(task_rows, phase) for phase in phases], rotation=30, ha="right")
        ax.set_ylim(0.0, y_upper)
        ax.set_xlim(-0.25, max(0.25, len(phases) - 0.75))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
        ax.tick_params(axis="x", pad=1)

    for ax in axes_flat[len(task_ids):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean IoU with GT union mask")
    legend: dict[str, Any] = {}
    for ax in axes_flat[: len(task_ids)]:
        handles, labels = ax.get_legend_handles_labels()
        legend.update({label: handle for handle, label in zip(handles, labels)})
    if legend:
        fig.legend(
            list(legend.values()),
            list(legend),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=max(1, min(4, len(legend))),
            frameon=False,
        )
    fig.suptitle(title or "Stage-wise attention localization on LIBERO", y=0.985, fontsize=8.8)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.13, hspace=0.82, wspace=0.42)
    return fig


def _phase_order(rows: list[AttentionIouRow]) -> list[str]:
    phases = list(dict.fromkeys(row.phase for row in rows))
    return sorted(phases, key=_natural_task_key)


def _phase_label(rows: list[AttentionIouRow], phase: str) -> str:
    for row in rows:
        if row.phase == phase:
            label = row.phase_label
            if label and label != phase:
                return label
            digits = "".join(ch for ch in phase if ch.isdigit())
            if digits:
                idx = int(digits)
                return f"Phase {idx}\n({_ordinal(idx)} sub-goal)"
            return label
    return phase.replace("_", " ")


def _natural_task_key(value: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(value))


def _metric_order(rows: list[AttentionIouRow]) -> list[str]:
    preferred = ("iou_top10_gt", "iou_top10", "iou_fixed_gt", "iou_fixedthr", "attn_mass")
    present = {row.metric for row in rows}
    ordered = [metric for metric in preferred if metric in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _metric_style(metric: str, idx: int) -> dict[str, Any]:
    if metric in {"iou_top10_gt", "iou_top10"}:
        return {"color": "#0072B2", "marker": "o", "linestyle": "-", "linewidth": 1.25, "markersize": 3.4}
    if metric in {"iou_fixed_gt", "iou_fixedthr"}:
        return {"color": "#D55E00", "marker": "s", "linestyle": "--", "linewidth": 1.0, "markersize": 3.0}
    palette = ("#009E73", "#CC79A7", "#4D4D4D")
    return {"color": palette[idx % len(palette)], "marker": "o", "linestyle": "-", "linewidth": 1.1, "markersize": 3.0}


def _metric_label(metric: str) -> str:
    labels = {
        "iou_top10_gt": "Top-10 patch IoU",
        "iou_top10": "Top-10 patch IoU",
        "iou_fixed_gt": "Fixed-threshold IoU",
        "iou_fixedthr": "Fixed-threshold IoU",
        "attn_mass": "Attention mass",
    }
    return labels.get(metric, metric.replace("_", " "))


def _metric_matches(actual: str, requested: str) -> bool:
    aliases = {
        "iou_top10": {"iou_top10", "iou_top10_gt"},
        "iou_top10_gt": {"iou_top10", "iou_top10_gt"},
        "iou_fixedthr": {"iou_fixedthr", "iou_fixed_gt"},
        "iou_fixed_gt": {"iou_fixedthr", "iou_fixed_gt"},
    }
    return actual in aliases.get(requested, {requested})


def _task_title(task_id: str, rows: list[AttentionIouRow]) -> str:
    instruction = next((row.instruction for row in rows if row.instruction), "")
    status_row = next((row for row in rows if row.total_steps is not None or row.success), None)
    suffix = ""
    if status_row is not None:
        pieces = []
        if status_row.total_steps is not None:
            pieces.append(f"{status_row.total_steps} steps")
        if status_row.success:
            pieces.append(_success_label(status_row.success))
        if pieces:
            suffix = f" ({', '.join(pieces)})"
    if instruction:
        return f"Task {task_id}{suffix}\n{_short_instruction(instruction)}"
    return f"Task {task_id}{suffix}"


def _short_instruction(text: str, width: int = 34, max_lines: int = 2) -> str:
    wrapped = textwrap.wrap(text, width=width)
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)
    return "\n".join(wrapped[:max_lines]) + "..."


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _success_label(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "success", "succeeded"}:
        return "success"
    if lowered in {"0", "false", "no", "failed", "failure"}:
        return "failed"
    return value.strip()
