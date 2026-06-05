"""Public Stage 2 knockout job manifest presets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vla_trace.io import write_json
from vla_trace.io.registry import DEFAULT_NUM_LAYERS, normalize_dataset, normalize_model
from vla_trace.knockout.specs import expand_layer_window


def build_standard_knockout_manifest(
    *,
    model: str,
    dataset: str,
    output_path: str | Path | None = None,
    window_size: int = 5,
    num_layers: int | None = None,
    trials: int | None = None,
) -> dict[str, Any]:
    """Build the standard layerwise/all-layer knockout job matrix.

    The returned manifest contains only portable job metadata and relative run
    tags. Users wire each job into their local LIBERO evaluator with their own
    checkpoint/data paths.
    """
    family = normalize_model(model)
    dataset_name = normalize_dataset(dataset)
    layers_n = int(num_layers or DEFAULT_NUM_LAYERS[family])
    layerwise_settings = _layerwise_settings(family)
    jobs: list[dict[str, Any]] = []
    for center in range(layers_n):
        layers = expand_layer_window(layers_n, (center,), window_size)
        for setting in layerwise_settings:
            jobs.append(
                {
                    **setting,
                    "job_type": "layerwise",
                    "model": family,
                    "dataset": dataset_name,
                    "center_layer": center,
                    "window_size": window_size,
                    "layers": list(layers),
                    "setting": f"{family}_layerwise_{setting['setting_suffix']}_window{window_size}",
                    "tag": f"{family}_layerwise_{setting['setting_suffix']}_window{window_size}/layer{center}_w{window_size}",
                }
            )
    for setting in _all_layer_settings(family):
        jobs.append(
            {
                **setting,
                "job_type": "all_layers",
                "model": family,
                "dataset": dataset_name,
                "center_layer": None,
                "window_size": None,
                "layers": list(range(layers_n)) if setting["mode"] != "baseline" else [],
                "setting": f"{family}_all_layers_{setting['setting_suffix']}",
                "tag": f"{family}_all_layers_{setting['setting_suffix']}",
            }
        )
    manifest = {
        "status": "ok",
        "preset": "standard_layerwise",
        "model": family,
        "dataset": dataset_name,
        "num_layers": layers_n,
        "window_size": window_size,
        "trials": trials,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    if output_path:
        write_json(output_path, manifest)
    return manifest


def _layerwise_settings(family: str) -> list[dict[str, Any]]:
    if family == "openvla":
        return [
            _job("no_image", "prefill", "all", "prefill_no_image"),
            _job("no_text", "generation", "all", "generation_no_text"),
            _job("no_image", "generation", "all", "generation_no_image"),
            _job("no_text", "generation", "newline_only", "generation_drop_newline"),
            _job("no_text", "generation", "exclude_newline", "generation_drop_spt_keep_newline"),
            _job(
                "openvla_prefill_no_image__generation_no_text",
                "both",
                "all",
                "prefill_no_image__generation_no_text",
            ),
            _job(
                "openvla_prefill_no_image__generation_no_image",
                "both",
                "all",
                "prefill_no_image__generation_no_image",
            ),
        ]
    if family == "pi05":
        return [
            _job("no_vl", "prefill", "all", "prefill_no_vl"),
            _job("no_text", "generation", "all", "generation_no_text"),
            _job("no_image", "generation", "all", "generation_no_image"),
            _job("no_vl", "generation", "instruction", "generation_keep_bos_newline_only"),
            _job("no_text", "generation", "bos_newline", "generation_drop_bos_newline"),
            _job("no_text", "generation", "instruction", "generation_drop_task_instruction"),
            _job(
                "pi05_prefill_no_vl__generation_no_text",
                "both",
                "all",
                "prefill_no_vl__generation_no_text",
            ),
            _job(
                "pi05_prefill_no_vl__generation_no_image",
                "both",
                "all",
                "prefill_no_vl__generation_no_image",
            ),
        ]
    raise ValueError(f"Unsupported knockout family: {family}")


def _all_layer_settings(family: str) -> list[dict[str, Any]]:
    if family == "openvla":
        return [
            _job("baseline", "generation", "all", "baseline"),
            _job(
                "openvla_prefill_no_image__generation_no_text",
                "both",
                "all",
                "prefill_no_image__generation_no_text",
            ),
            _job(
                "openvla_prefill_no_image__generation_no_image",
                "both",
                "all",
                "prefill_no_image__generation_no_image",
            ),
        ]
    if family == "pi05":
        return [
            _job("baseline", "generation", "all", "baseline"),
            _job(
                "pi05_prefill_no_vl__generation_no_text",
                "both",
                "all",
                "prefill_no_vl__generation_no_text",
            ),
            _job(
                "pi05_prefill_no_vl__generation_no_image",
                "both",
                "all",
                "prefill_no_vl__generation_no_image",
            ),
        ]
    raise ValueError(f"Unsupported knockout family: {family}")


def _job(mode: str, phase: str, text_scope: str, suffix: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "phase": phase,
        "text_scope": text_scope,
        "setting_suffix": suffix,
    }
