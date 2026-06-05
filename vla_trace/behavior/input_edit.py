"""Input-edit manifest generation for Stage 3 semantic probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_EDIT_TYPES = frozenset({"instruction_replace", "object_replace", "attribute_replace", "scene_replace"})


def build_input_edit_record(
    *,
    edit_id: str,
    task_id: str | int,
    edit_type: str,
    base_instruction: str,
    edited_instruction: str,
    target_object: str | None = None,
    replacement_object: str | None = None,
    target_attributes: dict[str, Any] | None = None,
    expected_shift: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if edit_type not in SUPPORTED_EDIT_TYPES:
        raise ValueError(f"Unsupported input edit type: {edit_type}")
    if not edit_id:
        raise ValueError("edit_id must not be empty")
    if not base_instruction or not edited_instruction:
        raise ValueError("base_instruction and edited_instruction are required")
    return {
        "edit_id": str(edit_id),
        "task_id": task_id,
        "edit_type": edit_type,
        "base_instruction": base_instruction,
        "edited_instruction": edited_instruction,
        "target_object": target_object or "",
        "replacement_object": replacement_object or "",
        "target_attributes": target_attributes or {},
        "expected_shift": expected_shift or "",
        "metadata": metadata or {},
    }


def build_records_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        key: config[key]
        for key in ("model", "dataset", "benchmark", "model_path", "data_root")
        if key in config
    }
    records = []
    edits = config.get("edits")
    if not isinstance(edits, list):
        raise ValueError("Input edit config must contain an `edits` list")
    for index, item in enumerate(edits):
        payload = dict(item)
        metadata = {**common, **dict(payload.pop("metadata", {}))}
        records.append(
            build_input_edit_record(
                edit_id=str(payload.pop("edit_id", f"edit_{index:04d}")),
                task_id=payload.pop("task_id", config.get("task_id", "")),
                edit_type=str(payload.pop("edit_type", config.get("edit_type", "instruction_replace"))),
                base_instruction=str(payload.pop("base_instruction", config.get("base_instruction", ""))),
                edited_instruction=str(payload.pop("edited_instruction", "")),
                target_object=payload.pop("target_object", None),
                replacement_object=payload.pop("replacement_object", None),
                target_attributes=payload.pop("target_attributes", None),
                expected_shift=payload.pop("expected_shift", None),
                metadata={**metadata, **payload},
            )
        )
    return records


def write_input_edit_manifest(path: str | Path, records: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".jsonl":
        with target.open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(record, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
                handle.write("\n")
        return target
    with target.open("w", encoding="utf-8") as handle:
        json.dump({"edits": records}, handle, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True)
        handle.write("\n")
    return target


def summarize_input_edit_results(path: str | Path) -> dict[str, Any]:
    rows = _read_result_rows(path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("edit_type", "unknown")), []).append(row)
    summaries = []
    for edit_type, group in sorted(groups.items()):
        successes = [bool(row.get("success", row.get("instruction_followed", False))) for row in group]
        summaries.append(
            {
                "edit_type": edit_type,
                "n": len(group),
                "success_rate": sum(successes) / len(successes) if successes else 0.0,
            }
        )
    return {"n": len(rows), "groups": summaries}


def _read_result_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        rows: list[dict[str, Any]] = []
        for item in sorted(source.rglob("*.json")):
            rows.extend(_read_result_rows(item))
        for item in sorted(source.rglob("*.jsonl")):
            rows.extend(_read_result_rows(item))
        return rows
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [dict(item) for item in payload["results"]]
    if isinstance(payload, dict):
        return [payload]
    return []
