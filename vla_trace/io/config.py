"""Config loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file."""
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config at {config_path}, got {type(payload).__name__}")
    return payload


def resolve_config_path(path: str | Path) -> Path:
    """Resolve public config paths from cwd, repo root, or package data."""
    config_path = Path(path)
    if config_path.exists() or config_path.is_absolute():
        return config_path
    repo_candidate = Path(__file__).resolve().parents[2] / config_path
    if repo_candidate.exists():
        return repo_candidate
    package_candidate = Path(__file__).resolve().parents[1] / config_path
    if package_candidate.exists():
        return package_candidate
    return config_path


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON artifact with stable formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return output
