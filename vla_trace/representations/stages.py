"""Multi-checkpoint representation collection plans for Stage 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from vla_trace.representations.extraction import collect_representation_bank_from_config


DEFAULT_STAGE_DESCRIPTIONS = {
    "C0": "pretrained VLM or action-free base model",
    "C1": "action-pretrained VLA checkpoint",
    "C2": "task- or benchmark-finetuned VLA checkpoint",
}


def build_stage_collection_plan(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve C0/C1/C2-style collection jobs without running model code."""

    stages = _resolve_stage_specs(cfg)
    jobs = []
    for name, spec in stages.items():
        job_cfg = _stage_collect_config(cfg, name, spec)
        missing = _missing_inputs(job_cfg)
        path_warnings = _path_warnings(job_cfg)
        jobs.append(
            {
                "stage": name,
                "description": spec.get("description") or DEFAULT_STAGE_DESCRIPTIONS.get(name, ""),
                "model_path": job_cfg.get("model_path"),
                "hidden_state_dir": job_cfg.get("hidden_state_dir"),
                "bank_output": job_cfg.get("output_path"),
                "adapter": job_cfg.get("adapter"),
                "checkpoint_name": name,
                "configured": not missing,
                "ready": not missing,
                "missing": missing,
                "path_warnings": path_warnings,
                "collect_command": _collect_command(job_cfg),
            }
        )
    return {
        "status": "planned",
        "command": "collect-repr-stages",
        "model": cfg.get("model"),
        "benchmark": cfg.get("benchmark"),
        "manifest_path": cfg.get("manifest_path"),
        "stage_semantics": DEFAULT_STAGE_DESCRIPTIONS,
        "n_stages": len(jobs),
        "jobs": jobs,
        "notes": [
            "C0/C1/C2 are user-provided checkpoint stages, not bundled artifacts.",
            "Each stage needs exported hidden states or an adapter that can run that checkpoint.",
        ],
    }


def collect_representation_stages_from_config(cfg: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Collect representation banks for every resolved stage."""

    plan = build_stage_collection_plan(cfg)
    if dry_run:
        return plan

    results = []
    stages = _resolve_stage_specs(cfg)
    for job in plan["jobs"]:
        if job["missing"]:
            missing = ", ".join(job["missing"])
            raise ValueError(f"Stage {job['stage']} is missing required inputs: {missing}")
        job_cfg = _stage_collect_config(cfg, job["stage"], stages[job["stage"]])
        results.append(collect_representation_bank_from_config(job_cfg))
    return {
        "status": "ok",
        "command": "collect-repr-stages",
        "n_stages": len(results),
        "results": results,
        "plan": plan,
    }


def _resolve_stage_specs(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = cfg.get("stages") or {}
    if isinstance(raw, list):
        stages = {str(item["name"]): dict(item) for item in raw}
    else:
        stages = {str(name): dict(spec or {}) for name, spec in dict(raw).items()}
    if not stages:
        stages = {name: {} for name in DEFAULT_STAGE_DESCRIPTIONS}
    return dict(sorted(stages.items(), key=lambda item: _stage_sort_key(item[0])))


def _stage_collect_config(cfg: Mapping[str, Any], stage_name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    job_cfg = {
        key: cfg[key]
        for key in (
            "model",
            "family",
            "benchmark",
            "manifest_path",
            "token_layout",
            "token_groups",
            "max_samples",
            "prompt_style",
            "adapter",
            "adapter_kwargs",
            "data_root",
        )
        if key in cfg
    }
    job_cfg.update({key: value for key, value in spec.items() if value is not None})
    job_cfg["checkpoint_name"] = stage_name
    if spec.get("model_path"):
        job_cfg["model_path"] = spec["model_path"]
    hidden_state_dir = spec.get("hidden_state_dir")
    if not hidden_state_dir and cfg.get("hidden_state_root"):
        hidden_state_dir = str(Path(str(cfg["hidden_state_root"])) / f"{stage_name}_hidden_states")
    if hidden_state_dir:
        job_cfg["hidden_state_dir"] = hidden_state_dir
    output_path = spec.get("output_path") or spec.get("bank_output")
    if not output_path and cfg.get("bank_root"):
        output_path = str(Path(str(cfg["bank_root"])) / f"{stage_name}_bank.npz")
    if output_path:
        job_cfg["output_path"] = output_path
    return job_cfg


def _missing_inputs(job_cfg: Mapping[str, Any]) -> list[str]:
    missing = []
    if not job_cfg.get("manifest_path"):
        missing.append("manifest_path")
    if not job_cfg.get("output_path"):
        missing.append("bank_output")
    if not job_cfg.get("hidden_state_dir") and not job_cfg.get("adapter"):
        missing.append("hidden_state_dir_or_adapter")
    return missing


def _path_warnings(job_cfg: Mapping[str, Any]) -> list[str]:
    warnings = []
    for key in ("manifest_path", "hidden_state_dir", "model_path"):
        value = job_cfg.get(key)
        if value and not Path(str(value)).exists():
            warnings.append(f"{key}_not_found")
    return warnings


def _collect_command(job_cfg: Mapping[str, Any]) -> list[str]:
    command = ["vla-trace", "collect-repr"]
    if job_cfg.get("model"):
        command += ["--model", _display_model_arg(str(job_cfg["model"]))]
    if job_cfg.get("benchmark"):
        command += ["--dataset", _display_dataset_arg(str(job_cfg["benchmark"]))]
    for key, flag in (
        ("manifest_path", "--manifest"),
        ("hidden_state_dir", "--hidden-state-dir"),
        ("adapter", "--adapter"),
        ("output_path", "--bank-output"),
        ("checkpoint_name", "--checkpoint-name"),
        ("model_path", "--model-path"),
        ("data_root", "--data-root"),
    ):
        if job_cfg.get(key):
            command += [flag, str(job_cfg[key])]
    for name, selector in dict(job_cfg.get("token_groups") or {}).items():
        command += ["--token-group", f"{name}={_selector_text(selector)}"]
    return command


def _display_model_arg(value: str) -> str:
    lowered = value.lower()
    if "pi05" in lowered or "pi0.5" in lowered:
        return "pi0.5"
    if "openvla" in lowered:
        return "OpenVLA"
    return value


def _display_dataset_arg(value: str) -> str:
    lowered = value.lower()
    for dataset in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        if dataset in lowered:
            return dataset
    return value


def _selector_text(selector: Any) -> str:
    if isinstance(selector, str):
        return selector
    if isinstance(selector, (list, tuple)) and len(selector) == 2:
        start = "" if selector[0] is None else str(selector[0])
        stop = "" if selector[1] is None else str(selector[1])
        return f"{start}:{stop}"
    if isinstance(selector, (list, tuple)):
        return ",".join(str(item) for item in selector)
    return str(selector)


def _stage_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("C") and name[1:].isdigit():
        return int(name[1:]), name
    return 999, name
