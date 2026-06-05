"""Config-facing Stage 1 CKA analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vla_trace.io import write_json
from vla_trace.representations.bank import RepresentationBank, load_representation_bank
from vla_trace.representations.cka import (
    checkpoint_drift_cka,
    cross_modal_cka_profile,
    layerwise_checkpoint_drift_cka,
    layerwise_cross_modal_cka,
    matched_layer_checkpoint_drift_summary,
)


def _load_bank_map(bank_paths: dict[str, str]) -> dict[str, RepresentationBank]:
    return {name: load_representation_bank(path) for name, path in bank_paths.items()}


def _write_result(cfg: dict[str, Any], filename: str, result: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: cfg[key]
        for key in ("model", "benchmark", "model_path", "data_root")
        if key in cfg
    }
    result = {**metadata, **result}
    output_dir = cfg.get("output_dir")
    if output_dir:
        output_path = Path(output_dir) / filename
        write_json(output_path, result)
        result = {**result, "output_path": str(output_path)}
    return result


def analyze_cross_modal_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run cross-modal CKA over saved representation banks.

    Expected config field:
        bank_paths: {C0: path/to/bank.json, C1: path/to/bank.npz, ...}
    """
    bank_paths = dict(cfg.get("bank_paths") or {})
    if not bank_paths:
        return _write_result(
            cfg,
            "cross_modal_cka_report.json",
            {
                "status": "missing-bank-paths",
                "analysis": "cross_modal",
                "message": "Set bank_paths to saved representation banks before running CKA.",
            },
        )

    banks = _load_bank_map(bank_paths)
    layerwise = bool(cfg.get("layerwise", False))
    if layerwise:
        profiles = {
            checkpoint: layerwise_cross_modal_cka(bank.arrays)
            for checkpoint, bank in banks.items()
        }
    else:
        profiles = {
            checkpoint: cross_modal_cka_profile(bank.arrays)
            for checkpoint, bank in banks.items()
        }
    result = {
        "status": "ok",
        "analysis": "cross_modal",
        "layerwise": layerwise,
        "profiles": profiles,
        "checkpoints": list(banks),
    }
    return _write_result(cfg, "cross_modal_cka_report.json", result)


def analyze_drift_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run checkpoint-drift CKA for one pooled view across saved banks."""
    bank_paths = dict(cfg.get("bank_paths") or {})
    view = str(cfg.get("view", "joint_pooled"))
    reference = cfg.get("reference")
    layerwise = bool(cfg.get("layerwise", False))
    if not bank_paths:
        return _write_result(
            cfg,
            "checkpoint_drift_cka_report.json",
            {
                "status": "missing-bank-paths",
                "analysis": "checkpoint_drift",
                "message": "Set bank_paths to saved representation banks before running drift CKA.",
            },
        )

    banks = _load_bank_map(bank_paths)
    if layerwise:
        names = list(banks)
        reference_name = str(reference or names[0])
        if reference_name not in banks:
            raise KeyError(f"unknown reference checkpoint: {reference_name}")
        profiles = {
            name: layerwise_checkpoint_drift_cka(banks[reference_name].arrays, bank.arrays, view=view)
            for name, bank in banks.items()
        }
        summary_views = cfg.get("summary_views", ("vision_pooled", "text_pooled", "joint_pooled"))
        result = {
            "status": "ok",
            "analysis": "checkpoint_drift",
            "layerwise": True,
            "view": view,
            "reference": reference_name,
            "profiles": profiles,
            "matched_layer_summary": matched_layer_checkpoint_drift_summary(
                {name: bank.arrays for name, bank in banks.items()},
                reference=reference_name,
                views=summary_views,
            ),
        }
        return _write_result(cfg, "checkpoint_drift_cka_report.json", result)

    checkpoints: dict[str, Any] = {}
    missing: list[str] = []
    for name, bank in banks.items():
        if view not in bank.arrays:
            missing.append(name)
        else:
            checkpoints[name] = bank.arrays[view]
    if missing:
        raise KeyError(f"View {view!r} missing from bank(s): {', '.join(missing)}")
    report = checkpoint_drift_cka(checkpoints, reference=reference)
    result = {
        "status": "ok",
        "analysis": "checkpoint_drift",
        "layerwise": False,
        "view": view,
        **report,
    }
    return _write_result(cfg, "checkpoint_drift_cka_report.json", result)
