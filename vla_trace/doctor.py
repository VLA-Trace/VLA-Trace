"""Repository and artifact readiness checks for community runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vla_trace.adapters import load_model_spec
from vla_trace.io import load_config, resolve_config_path
from vla_trace.representations.bank import load_representation_bank
from vla_trace.visualization.knockout import collect_knockout_results


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str


def run_doctor(
    *,
    model_config: str | Path | None = None,
    benchmark_config: str | Path | None = None,
    model_path: str | Path | None = None,
    data_root: str | Path | None = None,
    banks: dict[str, str] | None = None,
    attention_path: str | Path | None = None,
    masks_path: str | Path | None = None,
    results_path: str | Path | None = None,
    input_edits_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate local resources without recording private paths in repo files."""
    checks: list[DoctorCheck] = []
    if model_config:
        _check_model_config(checks, Path(model_config))
    else:
        checks.append(DoctorCheck("model_config", "warning", "No model config was provided. Use --model or --model-config."))

    if benchmark_config:
        _check_config_file(checks, "benchmark_config", Path(benchmark_config))
    else:
        checks.append(DoctorCheck("benchmark_config", "warning", "No benchmark config was provided. Use --dataset or --benchmark-config."))

    _check_optional_existing_path(checks, "model_path", model_path, "User checkpoint/model path")
    _check_optional_existing_path(checks, "data_root", data_root, "User dataset root")
    for name, path in sorted((banks or {}).items()):
        _check_bank(checks, name, Path(path))
    if attention_path:
        _check_array_artifact(checks, "attention", Path(attention_path), require_npz=False)
    if masks_path:
        _check_array_artifact(checks, "masks", Path(masks_path), require_npz=False)
    if attention_path and not masks_path:
        checks.append(DoctorCheck("attention_masks_pair", "warning", "Attention metrics need --masks in addition to --attention."))
    if masks_path and not attention_path:
        checks.append(DoctorCheck("attention_masks_pair", "warning", "Mask artifacts were provided without --attention."))
    if results_path:
        _check_results(checks, Path(results_path))
    if input_edits_path:
        _check_input_edits(checks, Path(input_edits_path))

    status = _overall_status(checks)
    return {
        "status": status,
        "checks": [asdict(check) for check in checks],
        "summary": {
            "ok": sum(check.status == "ok" for check in checks),
            "warning": sum(check.status == "warning" for check in checks),
            "error": sum(check.status == "error" for check in checks),
        },
    }


def _check_model_config(checks: list[DoctorCheck], path: Path) -> None:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        checks.append(DoctorCheck("model_config", "error", f"Model config does not exist: {path}"))
        return
    try:
        spec = load_model_spec(config_path)
    except Exception as exc:  # noqa: BLE001 - doctor reports readiness instead of raising first.
        checks.append(DoctorCheck("model_config", "error", f"Could not load model config {path}: {exc}"))
        return
    stages = ",".join(spec.supported_stages)
    checks.append(DoctorCheck("model_config", "ok", f"{spec.name} family={spec.family} stages={stages}"))


def _check_config_file(checks: list[DoctorCheck], name: str, path: Path) -> None:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        checks.append(DoctorCheck(name, "error", f"Config does not exist: {path}"))
        return
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        checks.append(DoctorCheck(name, "error", f"Could not load config {path}: {exc}"))
        return
    checks.append(DoctorCheck(name, "ok", f"Loaded {path} with keys: {', '.join(sorted(cfg)[:8])}"))


def _check_optional_existing_path(
    checks: list[DoctorCheck],
    name: str,
    path: str | Path | None,
    label: str,
) -> None:
    if not path:
        checks.append(DoctorCheck(name, "warning", f"{label} not provided; adapter-backed online runs will need it."))
        return
    source = Path(path)
    status = "ok" if source.exists() else "error"
    message = f"{label} exists: {source}" if source.exists() else f"{label} does not exist: {source}"
    checks.append(DoctorCheck(name, status, message))


def _check_bank(checks: list[DoctorCheck], name: str, path: Path) -> None:
    if not path.exists():
        checks.append(DoctorCheck(f"bank:{name}", "error", f"Bank does not exist: {path}"))
        return
    try:
        bank = load_representation_bank(path)
    except Exception as exc:  # noqa: BLE001
        checks.append(DoctorCheck(f"bank:{name}", "error", f"Could not load bank {path}: {exc}"))
        return
    if not bank.arrays:
        checks.append(DoctorCheck(f"bank:{name}", "error", f"Bank has no arrays: {path}"))
        return
    sample = next(iter(bank.arrays.values()))
    checks.append(DoctorCheck(f"bank:{name}", "ok", f"{len(bank.arrays)} arrays; first shape={list(sample.shape)}"))


def _check_array_artifact(checks: list[DoctorCheck], name: str, path: Path, *, require_npz: bool) -> None:
    if not path.exists():
        checks.append(DoctorCheck(name, "error", f"Artifact does not exist: {path}"))
        return
    if path.suffix not in {".npy", ".npz"}:
        checks.append(DoctorCheck(name, "error", f"Expected .npy or .npz artifact, got {path.suffix}"))
        return
    if require_npz and path.suffix != ".npz":
        checks.append(DoctorCheck(name, "error", f"Expected .npz artifact, got {path.suffix}"))
        return
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                if not payload.files:
                    checks.append(DoctorCheck(name, "error", f"NPZ artifact has no arrays: {path}"))
                    return
                first = payload[payload.files[0]]
                checks.append(DoctorCheck(name, "ok", f"{len(payload.files)} arrays; first shape={list(first.shape)}"))
        else:
            array = np.load(path, allow_pickle=False)
            checks.append(DoctorCheck(name, "ok", f"Array shape={list(array.shape)}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(DoctorCheck(name, "error", f"Could not read artifact {path}: {exc}"))


def _check_results(checks: list[DoctorCheck], path: Path) -> None:
    if not path.exists():
        checks.append(DoctorCheck("results", "error", f"Results path does not exist: {path}"))
        return
    try:
        rows = collect_knockout_results(path)
    except Exception as exc:  # noqa: BLE001
        checks.append(DoctorCheck("results", "error", f"Could not parse result artifacts {path}: {exc}"))
        return
    if not rows:
        checks.append(DoctorCheck("results", "warning", "No success-rate JSON rows found for plot-knockout."))
        return
    checks.append(DoctorCheck("results", "ok", f"Parsed {len(rows)} success-rate row(s)."))


def _check_input_edits(checks: list[DoctorCheck], path: Path) -> None:
    if not path.exists():
        checks.append(DoctorCheck("input_edits", "error", f"Input-edit manifest does not exist: {path}"))
        return
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("edits", payload if isinstance(payload, list) else [])
    except Exception as exc:  # noqa: BLE001
        checks.append(DoctorCheck("input_edits", "error", f"Could not parse input-edit manifest {path}: {exc}"))
        return
    if not rows:
        checks.append(DoctorCheck("input_edits", "error", f"No edits found in {path}"))
        return
    checks.append(DoctorCheck("input_edits", "ok", f"Found {len(rows)} edit record(s)."))


def _overall_status(checks: list[DoctorCheck]) -> str:
    if any(check.status == "error" for check in checks):
        return "error"
    if any(check.status == "warning" for check in checks):
        return "warning"
    return "ok"
