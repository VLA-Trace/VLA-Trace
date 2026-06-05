"""Manifest helpers for Stage 1 representation-bank collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_manifest(path: str | Path, *, max_samples: int | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")
    return target


def build_manifest_row(
    sample_idx: int,
    *,
    task_name: str,
    instruction: str,
    image_path: str | Path,
    task_id: str | int | None = None,
    hidden_states_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": f"{task_name}__{int(sample_idx):06d}",
        "task_name": str(task_name),
        "instruction": str(instruction).strip(),
        "image_path": str(image_path),
    }
    if task_id is not None:
        row["task_id"] = task_id
    if hidden_states_path is not None:
        row["hidden_states_path"] = str(hidden_states_path)
    if metadata:
        row["metadata"] = dict(metadata)
    return row


def export_libero_manifest(
    *,
    data_root: str | Path,
    data_mix: str,
    output_dir: str | Path,
    max_samples: int,
    image_size: int = 224,
    train: bool = True,
    shuffle_buffer_size: int = 500,
    per_task_quota: int | None = None,
    max_stream_reads: int | None = None,
) -> dict[str, Any]:
    """Export CKA-ready image samples and a manifest from a LIBERO RLDS dataset.

    This mirrors the paper extraction path while keeping the dependency
    optional. Users install LIBERO/VLM4VLA-compatible RLDS support locally and
    pass their own dataset root at runtime.
    """
    try:
        from PIL import Image
        from vlm4vla.data.base_openvla_dataset import RLDSDataset
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "export-libero-manifest requires PIL and a local LIBERO RLDS "
            "dataset loader compatible with vlm4vla.data.base_openvla_dataset.RLDSDataset."
        ) from exc

    dataset = RLDSDataset(
        data_root_dir=Path(data_root),
        data_mix=data_mix,
        image_size=image_size,
        chunk_action=False,
        frame_num=1,
        left_pad=False,
        window_sample="sliding",
        window_size=1,
        fwd_pred_next_n=1,
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=False,
    )

    out_root = Path(output_dir)
    images_dir = out_root / "images"
    manifest_path = out_root / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    if max_stream_reads is None:
        max_stream_reads = max(max_samples * 20, 50000)

    saved = 0
    for idx, batch in enumerate(dataset):
        if idx >= max_stream_reads or saved >= max_samples:
            break
        instruction = _decode_instruction(batch)
        task_key = instruction.strip().lower()
        if per_task_quota is not None and task_counts.get(task_key, 0) >= per_task_quota:
            continue
        image = _extract_primary_image(batch)
        image_path = images_dir / f"{saved:06d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.astype(np.uint8)).save(image_path)
        task_name = batch.get("dataset_name") or batch.get("dataset_id") or data_mix
        rows.append(
            build_manifest_row(
                saved,
                task_name=str(task_name),
                instruction=instruction,
                image_path=image_path,
                task_id=task_key,
            )
        )
        task_counts[task_key] = task_counts.get(task_key, 0) + 1
        saved += 1

    write_manifest(manifest_path, rows)
    return {
        "status": "ok",
        "images_dir": str(images_dir),
        "manifest_path": str(manifest_path),
        "n_samples": len(rows),
        "task_counts": task_counts,
    }


def _extract_primary_image(batch: dict[str, Any]) -> np.ndarray:
    image = np.asarray(batch["observation"]["image_primary"])
    if image.ndim == 4:
        return image[-1]
    if image.ndim == 3:
        return image
    raise ValueError(f"Unsupported image_primary shape: {image.shape}")


def _decode_instruction(batch: dict[str, Any]) -> str:
    instruction = batch["task"]["language_instruction"]
    if isinstance(instruction, bytes):
        return instruction.decode()
    return str(instruction)
