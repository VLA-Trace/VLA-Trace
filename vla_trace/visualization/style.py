"""Shared plotting style for public VLA-Trace figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PALETTE = (
    "#2F6B8F",
    "#C45A31",
    "#4F8A5B",
    "#7B5EA7",
    "#B47B20",
    "#4C6FBD",
    "#8F4C69",
)


def load_pyplot() -> Any:
    """Import matplotlib lazily so plotting remains an optional dependency."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
        raise RuntimeError(
            "Plotting requires matplotlib. Install with `python -m pip install -e '.[plot]'`."
        ) from exc
    return plt


def apply_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def apply_publication_style(plt: Any) -> None:
    """Apply the paper plotting contract used by the VLA-Trace figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.7,
            "axes.titlesize": 6.7,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: Any, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    return target


def save_figure_all(fig: Any, stem: str | Path, formats: tuple[str, ...] = ("png", "pdf", "svg")) -> list[Path]:
    target = Path(stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        output = target.with_suffix(f".{suffix.lstrip('.')}")
        fig.savefig(output)
        outputs.append(output)
    return outputs


def add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )


def annotate_heatmap(ax: Any, matrix: Any, *, fmt: str = ".2f") -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, format(float(value), fmt), ha="center", va="center", fontsize=6.5)
