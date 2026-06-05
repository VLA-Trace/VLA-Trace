"""Plot Stage 2 knockout evaluation artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from vla_trace.visualization.style import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    apply_style,
    load_pyplot,
    save_figure,
    save_figure_all,
)


LAYER_RE = re.compile(r"layer(?P<layer>\d+)(?:_w(?P<window>\d+))?")
DATASET_RE = re.compile(r"libero_(?:10|goal|object|spatial)")
SOURCE_GROUP_RE = re.compile(r"(?:^|[_-])(?:run|release)?[_-]?\d{3,8}(?:$|[_-])")
MODEL_NAMES = ("openvla_oft", "openvla", "pi05")
OFT_SETTING_ALIASES = {
    "drop_newline_only": "generation_drop_newline",
    "drop_task_instruction": "generation_drop_task_instruction",
    "keep_newline_only": "generation_keep_newline_only",
    "no_all_special": "generation_no_all_special",
    "no_bos": "generation_drop_bos",
    "no_bos_newline": "generation_drop_bos_newline",
    "no_image": "generation_no_image",
    "no_newline": "generation_drop_newline",
    "no_perception": "generation_no_perception",
    "no_proprio": "generation_no_proprio",
    "no_text": "generation_no_text",
    "no_text_full": "generation_no_text_full",
    "no_vision": "generation_no_vision",
    "no_vl": "prefill_no_vl",
    "prefill_no_vl": "prefill_no_vl",
}
InputPaths = Iterable[str | Path] | str | Path

DATASET_ORDER = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DATASET_LABELS = {
    "libero_10": "LIBERO-10",
    "libero_goal": "Goal",
    "libero_object": "Object",
    "libero_spatial": "Spatial",
}
MODEL_ORDER = ("pi05", "openvla", "openvla_oft")
MODEL_LABELS = {
    "pi05": r"$\pi$0.5",
    "openvla": "OpenVLA",
    "openvla_oft": "OpenVLA-OFT",
}
MODEL_COLORS = {
    "pi05": "#484878",
    "openvla": "#0F4D92",
    "openvla_oft": "#5F7D3A",
}
MAIN_SETTINGS = {
    "pi05": (
        ("Generation: no image", ("generation_no_image",), "Gen no image"),
        ("Generation: no text", ("generation_no_text",), "Gen no text"),
        ("Prefill: no V-L", ("prefill_no_vl",), "Prefill no V-L"),
        (
            "Combined: prefill no V-L + generation no image",
            ("prefill_no_vl__generation_no_image", "prefill_no_vl_generation_no_image"),
            "Comb: no V-L + no image",
        ),
        (
            "Combined: prefill no V-L + generation no text",
            ("prefill_no_vl__generation_no_text", "prefill_no_vl_generation_no_text"),
            "Comb: no V-L + no text",
        ),
    ),
    "openvla": (
        ("Generation: no image", ("generation_no_image",), "Gen no image"),
        ("Generation: no text", ("generation_no_text",), "Gen no text"),
        ("Prefill: no image", ("prefill_no_image",), "Prefill no image"),
        (
            "Combined: prefill no image + generation no image",
            ("prefill_no_image__generation_no_image", "prefill_no_image_generation_no_image"),
            "Comb: no image + no image",
        ),
        (
            "Combined: prefill no image + generation no text",
            ("prefill_no_image__generation_no_text", "prefill_no_image_generation_no_text"),
            "Comb: no image + no text",
        ),
    ),
    "openvla_oft": (
        ("Generation: no vision", ("generation_no_vision", "generation_no_image"), "Gen no vision"),
        ("Generation: no text", ("generation_no_text",), "Gen no text"),
        ("Generation: no text full", ("generation_no_text_full",), "Gen no text full"),
        ("Prefill: no V-L", ("prefill_no_vl",), "Prefill no V-L"),
        ("Generation: no proprio", ("generation_no_proprio",), "Gen no proprio"),
    ),
}


@dataclass(frozen=True)
class KnockoutResult:
    path: str
    model: str
    dataset: str
    setting: str
    canonical_setting: str
    layer: int | None
    window: int | None
    success_rate: float
    success_num: int | None
    test_num: int | None
    ci_low: float | None
    ci_high: float | None
    is_baseline: bool


def plot_knockout_results(
    inputs: InputPaths,
    output: str | Path,
    *,
    model: str | None = None,
    dataset: str | None = None,
    setting: str | None = None,
    title: str | None = None,
    source_data: str | Path | None = None,
) -> Path:
    rows = collect_knockout_results(inputs)
    rows = _filter_rows(rows, model=model, dataset=dataset, setting=setting)
    if not rows:
        raise ValueError("No knockout result rows found. Expected JSON files with success_rate or success_num/test_num.")
    if source_data:
        _write_csv(source_data, [asdict(row) for row in rows])

    plt = load_pyplot()
    apply_style(plt)
    layer_rows = [row for row in rows if row.layer is not None and not row.is_baseline]
    if layer_rows:
        fig = _plot_layerwise(plt, layer_rows, rows, title)
    else:
        fig = _plot_bars(plt, rows, title)

    target = save_figure(fig, output)
    plt.close(fig)
    return target


def plot_knockout_line_grid(
    inputs: InputPaths | None,
    output_dir: str | Path,
    *,
    selected_csv: str | Path | None = None,
    baseline_csv: str | Path | None = None,
    model: str | None = None,
    dataset: str | None = None,
    protocol: str = "window7",
    source_data_dir: str | Path | None = None,
) -> list[Path]:
    """Draw publication-style layerwise knockout line grids.

    This keeps all paths explicit. Users may pass either publication source-data CSVs or raw result
    JSON directories accepted by `plot-knockout`.
    """
    if selected_csv:
        rows = _read_dict_csv(selected_csv)
    elif inputs is not None:
        rows = _line_rows_from_results(collect_knockout_results(inputs))
    else:
        raise ValueError("plot-knockout-line requires either result inputs or --selected-csv")
    baseline_rows = _read_dict_csv(baseline_csv) if baseline_csv else _baseline_rows_from_line_rows(rows)

    wanted_model = _normalize_filter(model) if model else None
    wanted_dataset = dataset.lower().replace("-", "_").replace("libero10", "libero_10") if dataset else None
    model_order = [name for name in MODEL_ORDER if wanted_model is None or _normalize_filter(name) == wanted_model]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_data_dir) if source_data_dir else None
    if source_root:
        source_root.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    plotted_all: list[dict[str, Any]] = []
    plt = load_pyplot()
    apply_publication_style(plt)
    for model_name in model_order:
        model_rows = [
            row for row in rows
            if _normalize_filter(str(row.get("model", ""))) == _normalize_filter(model_name)
            and (wanted_dataset is None or str(row.get("dataset", "")).lower() == wanted_dataset)
        ]
        if not model_rows:
            continue
        fig, plotted = _plot_line_grid_for_model(plt, model_name, model_rows, baseline_rows, protocol=protocol)
        outputs.extend(save_figure_all(fig, output_root / f"fig_line_layerwise_{model_name}"))
        plt.close(fig)
        plotted_all.extend(plotted)

    if not outputs:
        raise ValueError("No knockout rows matched the requested publication line-grid filters")
    if source_root:
        _write_csv(source_root / "main_line_layerwise_selected.csv", plotted_all)
        _write_csv(source_root / "baselines.csv", baseline_rows)
    return outputs


def collect_knockout_results(inputs: InputPaths) -> list[KnockoutResult]:
    rows: list[KnockoutResult] = []
    for path in _iter_json_paths(inputs):
        rows.extend(_read_result_rows(path))
    return sorted(rows, key=lambda row: (row.model, row.dataset, row.canonical_setting, row.layer or -1, row.path))


def _iter_json_paths(inputs: InputPaths) -> Iterable[Path]:
    items = (inputs,) if isinstance(inputs, (str, Path)) else inputs
    for item in items:
        path = Path(item)
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
        elif path.is_file() and path.suffix == ".json":
            yield path


def _read_result_rows(path: Path) -> list[KnockoutResult]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [_row for item in payload for _row in _payload_to_rows(path, item, {})]
    if not isinstance(payload, dict):
        return []
    common = {
        key: payload[key]
        for key in ("model", "dataset", "benchmark", "phase", "mode", "setting")
        if key in payload
    }
    if isinstance(payload.get("results"), list):
        return [_row for item in payload["results"] for _row in _payload_to_rows(path, item, common)]
    return _payload_to_rows(path, payload, common)


def _payload_to_rows(path: Path, payload: Any, common: dict[str, Any]) -> list[KnockoutResult]:
    if not isinstance(payload, dict):
        return []
    doc = {**common, **payload}
    rate = _success_rate_percent(doc)
    if rate is None:
        return []

    success_num = _optional_int(doc.get("success_num"))
    test_num = _optional_int(doc.get("test_num"))
    ci_low, ci_high = _ci_bounds(doc, success_num, test_num)
    setting = _infer_setting(path, doc)
    layer = _infer_layer(path, doc)
    window = _infer_window(path, doc)
    canonical = canonical_setting(setting)
    return [
        KnockoutResult(
            path=str(path),
            model=_infer_model(path, doc),
            dataset=_infer_dataset(path, doc),
            setting=setting,
            canonical_setting=canonical,
            layer=layer,
            window=window,
            success_rate=rate,
            success_num=success_num,
            test_num=test_num,
            ci_low=ci_low,
            ci_high=ci_high,
            is_baseline="baseline" in canonical,
        )
    ]


def _success_rate_percent(payload: dict[str, Any]) -> float | None:
    success_num = _optional_int(payload.get("success_num"))
    test_num = _optional_int(payload.get("test_num"))
    if "success_rate" in payload:
        rate = float(payload["success_rate"])
        if rate <= 1.0:
            return rate * 100.0
        return rate
    if success_num is not None and test_num:
        return success_num / test_num * 100.0
    return None


def _ci_bounds(payload: dict[str, Any], success_num: int | None, test_num: int | None) -> tuple[float | None, float | None]:
    if "ci_low" in payload and "ci_high" in payload:
        low = float(payload["ci_low"])
        high = float(payload["ci_high"])
        if low <= 1.0 and high <= 1.0:
            return low * 100.0, high * 100.0
        return low, high
    if success_num is None or test_num is None:
        return None, None
    return wilson_interval(success_num, test_num)


def wilson_interval(success_num: int, test_num: int, z: float = 1.96) -> tuple[float, float]:
    if test_num <= 0:
        return float("nan"), float("nan")
    p = success_num / test_num
    denom = 1 + z**2 / test_num
    centre = p + z**2 / (2 * test_num)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * test_num)) / test_num)
    return (centre - spread) / denom * 100.0, (centre + spread) / denom * 100.0


def canonical_setting(setting: str) -> str:
    name = setting.replace("-centercrop", "")
    for prefix in ("openvla_oft_", "openvla_", "pi05_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    name = re.sub(r"^(layerwise|all_layers)_", "", name)
    return re.sub(r"_window\d+$", "", name)


def _infer_model(path: Path, payload: dict[str, Any]) -> str:
    raw = str(payload.get("model") or payload.get("family") or "")
    lowered = raw.lower().replace(".", "")
    if "pi05" in lowered or "pi0" in lowered:
        return "pi05"
    if "openvla_oft" in lowered or "oft" in lowered:
        return "openvla_oft"
    if "openvla" in lowered:
        return "openvla"
    if _is_oft_path(path):
        return "openvla_oft"
    path_text = str(path).lower()
    for model in MODEL_NAMES:
        if model in path_text:
            return model
    return "unknown"


def _infer_dataset(path: Path, payload: dict[str, Any]) -> str:
    for key in ("dataset", "benchmark"):
        value = str(payload.get(key) or "").lower().replace("-", "_")
        value = value.replace("libero10", "libero_10")
        match = DATASET_RE.search(value)
        if match:
            return match.group(0)
    path_text = str(path).lower().replace("-", "_").replace("libero10", "libero_10")
    match = DATASET_RE.search(path_text)
    return match.group(0) if match else "unknown"


def _infer_setting(path: Path, payload: dict[str, Any]) -> str:
    if payload.get("setting"):
        return str(payload["setting"])
    phase = payload.get("phase")
    mode = payload.get("mode")
    if not phase and isinstance(payload.get("spec"), dict):
        phase = payload["spec"].get("phase")
        mode = payload["spec"].get("mode")
    if phase and mode:
        return f"{phase}_{mode}"
    if _is_oft_path(path):
        return _infer_oft_setting(path, _infer_layer(path, payload), _infer_window(path, payload))
    for part in reversed(path.parts):
        clean = part.replace("-centercrop", "")
        if clean.startswith(("openvla_oft_", "openvla_", "pi05_")):
            return clean
    return "baseline" if payload.get("is_baseline") else path.parent.name


def _infer_layer(path: Path, payload: dict[str, Any]) -> int | None:
    if payload.get("layer") is not None:
        return int(payload["layer"])
    layers = payload.get("layers")
    if isinstance(layers, list) and len(layers) == 1:
        return int(layers[0])
    if isinstance(payload.get("spec"), dict):
        spec_layers = payload["spec"].get("layers")
        if isinstance(spec_layers, list) and len(spec_layers) == 1:
            return int(spec_layers[0])
    match = LAYER_RE.search(str(path))
    return int(match.group("layer")) if match else None


def _infer_window(path: Path, payload: dict[str, Any]) -> int | None:
    if payload.get("window") is not None:
        return int(payload["window"])
    match = LAYER_RE.search(str(path))
    if match and match.group("window"):
        return int(match.group("window"))
    setting = str(payload.get("setting") or "")
    match = re.search(r"window(?P<window>\d+)", setting)
    return int(match.group("window")) if match else None


def _infer_crop(path: Path) -> str:
    return "centercrop" if any("centercrop" in part for part in path.parts) else ""


def _is_oft_path(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    return "oft" in parts or any("openvla-7b-oft" in part or "oft-finetuned" in part for part in parts)


def _infer_oft_setting(path: Path, layer: int | None, window: int | None) -> str:
    parent = path.parent.name.replace("-centercrop", "")
    if LAYER_RE.search(parent):
        raw = path.parent.parent.name.replace("-centercrop", "")
    elif "openvla-7b-oft" in parent or "oft-finetuned" in parent:
        raw = "baseline"
    else:
        raw = parent
    if raw.startswith("openvla_"):
        raw = raw.removeprefix("openvla_")
    raw = OFT_SETTING_ALIASES.get(raw, raw)
    if raw == "baseline":
        return "openvla_oft_baseline"
    if layer is not None or "layerwise" in {part.lower() for part in path.parts}:
        suffix = f"_window{window}" if window is not None else ""
        return f"openvla_oft_layerwise_{raw}{suffix}"
    return f"openvla_oft_all_layers_{raw}"


def _infer_source_group(path: Path) -> str:
    for part in path.parts:
        if "knockout" in part:
            return part
    for part in path.parts:
        if SOURCE_GROUP_RE.search(part):
            return part
    return ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _filter_rows(
    rows: list[KnockoutResult],
    *,
    model: str | None,
    dataset: str | None,
    setting: str | None,
) -> list[KnockoutResult]:
    out = rows
    if model:
        wanted = _normalize_filter(model)
        out = [row for row in out if _normalize_filter(row.model) == wanted]
    if dataset:
        wanted_dataset = dataset.lower().replace("-", "_").replace("libero10", "libero_10")
        out = [row for row in out if row.dataset == wanted_dataset]
    if setting:
        wanted_setting = setting.lower()
        out = [row for row in out if wanted_setting in row.canonical_setting.lower()]
    return out


def _normalize_filter(value: str) -> str:
    return value.lower().replace(".", "").replace("-", "_")


def _plot_layerwise(plt: Any, layer_rows: list[KnockoutResult], all_rows: list[KnockoutResult], title: str | None) -> Any:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    groups: dict[tuple[str, str, str, int | None], list[KnockoutResult]] = {}
    for row in layer_rows:
        key = (row.model, row.dataset, row.canonical_setting, row.window)
        groups.setdefault(key, []).append(row)

    single_context = len({(row.model, row.dataset) for row in layer_rows}) == 1
    for idx, (key, rows) in enumerate(sorted(groups.items())):
        model, dataset, setting, window = key
        rows = sorted(rows, key=lambda row: int(row.layer or 0))
        label = _setting_label(model, dataset, setting, window, compact=single_context)
        color = PALETTE[idx % len(PALETTE)]
        xs = [int(row.layer or 0) for row in rows]
        ys = [row.success_rate for row in rows]
        ax.plot(xs, ys, color=color, marker="o", linewidth=1.3, markersize=3.0, label=label)
        if all(row.ci_low is not None and row.ci_high is not None for row in rows):
            lows = [float(row.ci_low) for row in rows]
            highs = [float(row.ci_high) for row in rows]
            ax.fill_between(xs, lows, highs, color=color, alpha=0.13, linewidth=0)

    baselines = _baseline_by_model_dataset(all_rows)
    if single_context and baselines:
        baseline = next(iter(baselines.values()))
        ax.axhline(baseline, color="#5E5E5E", linestyle=(0, (4, 2)), linewidth=0.9, label="baseline")

    ax.set_title(title or "Layer-wise Knockout Evaluation")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False, ncols=1)
    fig.tight_layout()
    return fig


def _plot_bars(plt: Any, rows: list[KnockoutResult], title: str | None) -> Any:
    rows = sorted(rows, key=lambda row: (row.model, row.dataset, row.canonical_setting))
    fig, ax = plt.subplots(figsize=(max(4.8, len(rows) * 0.65), 3.5))
    labels = [
        _setting_label(row.model, row.dataset, row.canonical_setting, row.window, compact=False)
        for row in rows
    ]
    values = [row.success_rate for row in rows]
    colors = [PALETTE[idx % len(PALETTE)] for idx in range(len(rows))]
    ax.bar(range(len(rows)), values, color=colors, width=0.72)
    ax.set_title(title or "Knockout Evaluation")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(rows)), labels=labels, rotation=35, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    return fig


def _plot_line_grid_for_model(
    plt: Any,
    model: str,
    selected_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    protocol: str,
) -> tuple[Any, list[dict[str, Any]]]:
    specs = MAIN_SETTINGS.get(model) or ()
    datasets = [dataset for dataset in DATASET_ORDER if any(str(row.get("dataset")) == dataset for row in selected_rows)]
    if not datasets:
        datasets = list(DATASET_ORDER)
    rows_n = max(1, len(specs))
    cols_n = max(1, len(datasets))
    fig, axes = plt.subplots(
        rows_n,
        cols_n,
        figsize=(max(3.2, cols_n * 2.7), max(2.4, rows_n * 2.02)),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    try:
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, hspace=0.055, wspace=0.04)
    except AttributeError:
        pass

    color = MODEL_COLORS.get(model, PALETTE[0])
    panel_iter = iter(string.ascii_lowercase)
    plotted_rows: list[dict[str, Any]] = []
    for row_idx, (label, keys, short_label) in enumerate(specs):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            add_panel_label(ax, next(panel_iter))
            if row_idx == 0:
                ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}\n{protocol}", fontsize=8.2, pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{short_label or label}\nSuccess rate (%)", fontsize=7.2)
            data = _select_line_setting_rows(selected_rows, model, dataset, label, keys, protocol)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
            ax.tick_params(axis="both", labelsize=6.8)
            if not data:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#A8A8A8")
                ax.set_xticks([])
                continue

            data = sorted(data, key=lambda row: int(float(row.get("layer", 0))))
            plotted_rows.extend(data)
            xs = [int(float(row["layer"])) for row in data]
            ys = [float(row["success_rate"]) for row in data]
            lows = [float(row.get("ci_low", row["success_rate"])) for row in data]
            highs = [float(row.get("ci_high", row["success_rate"])) for row in data]
            ax.fill_between(xs, lows, highs, color=color, alpha=0.14, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=1.65, marker="o", markersize=3.1, markeredgewidth=0)
            base = _publication_baseline(baseline_rows, model, dataset)
            if base is not None:
                ax.axhline(base, color="#5F5F5F", linestyle=(0, (4, 2.4)), linewidth=0.8, alpha=0.8)
                ax.text(0.985, min(base / 105 + 0.01, 0.98), "base", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#5F5F5F")
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            ax.set_xticks(_layer_ticks(min(xs), max(xs)))
            if row_idx == rows_n - 1:
                ax.set_xlabel("Layer", fontsize=7.2)
            else:
                ax.set_xticklabels([])

    fig.suptitle(f"{MODEL_LABELS.get(model, model)} layer-wise knockout vulnerability", fontsize=10.2, y=1.012)
    return fig, plotted_rows


def _select_line_setting_rows(
    rows: list[dict[str, Any]],
    model: str,
    dataset: str,
    label: str,
    keys: tuple[str, ...],
    protocol: str,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if _normalize_filter(str(row.get("model", ""))) != _normalize_filter(model):
            continue
        if str(row.get("dataset", "")) != dataset:
            continue
        row_protocol = str(row.get("figure_protocol") or row.get("protocol") or _protocol_from_window(row.get("window")) or protocol)
        if row_protocol != protocol:
            continue
        row_label = str(row.get("main_setting_label") or "")
        setting = str(row.get("canonical_setting") or row.get("setting") or "")
        if row_label == label or any(key in setting for key in keys):
            normalized = dict(row)
            normalized["main_setting_label"] = label
            normalized["figure_protocol"] = protocol
            selected.append(normalized)
    return selected


def _line_rows_from_results(results: list[KnockoutResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if result.layer is None or result.is_baseline:
            continue
        model = _normalize_publication_model(result.model)
        crop = _infer_crop(Path(result.path))
        is_layerwise = result.layer is not None
        is_all_layers = result.layer is None and not result.is_baseline
        rows.append(
            {
                "path": result.path,
                "source_root": "",
                "source_group": _infer_source_group(Path(result.path)),
                "source_priority": 0,
                "model": model,
                "dataset": result.dataset,
                "setting": result.setting,
                "canonical_setting": result.canonical_setting,
                "layer": result.layer,
                "window": result.window or "",
                "protocol": _protocol_from_window(result.window),
                "crop": crop,
                "figure_protocol": _protocol_from_window(result.window),
                "success_rate": result.success_rate,
                "success_rate_fraction": result.success_rate / 100.0,
                "success_num": result.success_num or "",
                "test_num": result.test_num or "",
                "ci_low": result.ci_low if result.ci_low is not None else result.success_rate,
                "ci_high": result.ci_high if result.ci_high is not None else result.success_rate,
                "is_layerwise": is_layerwise,
                "is_baseline": False,
                "is_all_layers": is_all_layers,
                "intervention_scope": "layerwise" if is_layerwise else "all_layers",
                "main_setting_label": _main_setting_label(model, result.canonical_setting),
            }
        )
    return rows


def _baseline_rows_from_line_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        setting = str(row.get("canonical_setting") or row.get("setting") or "")
        if "baseline" in setting:
            grouped.setdefault((str(row.get("model")), str(row.get("dataset"))), []).append(float(row["success_rate"]))
    return [
        {
            "model": model,
            "dataset": dataset,
            "baseline_success_rate": median(values),
            "baseline_sources": len(values),
            "baseline_min": min(values),
            "baseline_max": max(values),
        }
        for (model, dataset), values in sorted(grouped.items())
        if values
    ]


def _publication_baseline(rows: list[dict[str, Any]], model: str, dataset: str) -> float | None:
    for row in rows:
        if _normalize_filter(str(row.get("model", ""))) == _normalize_filter(model) and str(row.get("dataset")) == dataset:
            value = row.get("baseline_success_rate") or row.get("success_rate")
            if value not in (None, ""):
                return float(value)
    return None


def _main_setting_label(model: str, setting: str) -> str:
    for label, keys, _short in MAIN_SETTINGS.get(model, ()):
        if any(key in setting for key in keys):
            return label
    return setting.replace("_", " ").title()


def _normalize_publication_model(model: str) -> str:
    normalized = _normalize_filter(model)
    if normalized == "pi0_5":
        return "pi05"
    if normalized in {"openvla_oft", "openvla", "pi05"}:
        return normalized
    return normalized


def _protocol_from_window(window: Any) -> str:
    if window in (None, ""):
        return "non_window"
    try:
        return f"window{int(window)}"
    except (TypeError, ValueError):
        return str(window)


def _layer_ticks(x_min: int, x_max: int) -> list[int]:
    if x_max <= 18:
        return list(range(x_min, x_max + 1, 4)) or [x_min]
    ticks = list(range(x_min, x_max + 1, 8))
    if not ticks or ticks[-1] != x_max:
        ticks.append(x_max)
    return ticks


def _read_dict_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [_normalize_csv_row(row) for row in csv.DictReader(handle)]


def _normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if normalized.get("model"):
        normalized["model"] = _normalize_publication_model(str(normalized["model"]))
    if normalized.get("dataset"):
        normalized["dataset"] = str(normalized["dataset"]).lower().replace("-", "_").replace("libero10", "libero_10")
    return normalized


def _baseline_by_model_dataset(rows: list[KnockoutResult]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.is_baseline:
            grouped.setdefault((row.model, row.dataset), []).append(row.success_rate)
    return {key: median(values) for key, values in grouped.items()}


def _setting_label(model: str, dataset: str, setting: str, window: int | None, *, compact: bool) -> str:
    base = setting.replace("_", " ")
    if window is not None:
        base = f"{base} w{window}"
    if compact:
        return base
    return f"{model}/{dataset}/{base}"


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
