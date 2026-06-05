from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vla_trace.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_collect_repr_stages_dry_run_reports_c0_c1_c2(capsys):
    assert (
        main(
            [
                "collect-repr-stages",
                str(ROOT / "configs/experiments/openvla_libero_stage_banks.yaml"),
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    assert [job["stage"] for job in payload["jobs"]] == ["C0", "C1", "C2"]
    assert "not bundled artifacts" in payload["notes"][0]
    assert payload["jobs"][0]["configured"] is True
    assert "hidden_state_dir_not_found" in payload["jobs"][0]["path_warnings"]


def test_collect_repr_stages_builds_three_banks(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"sample_id": "s0", "image_path": "images/0.png", "instruction": "pick object"}]
    manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    hidden_root = tmp_path / "hidden"
    hidden_root.mkdir()
    for stage, offset in [("C0", 0), ("C1", 10), ("C2", 20)]:
        stage_dir = hidden_root / f"{stage}_hidden_states"
        stage_dir.mkdir()
        hidden = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4) + offset
        np.savez(stage_dir / "s0.npz", hidden_states=hidden)

    report = tmp_path / "report.json"
    bank_root = tmp_path / "banks"
    assert (
        main(
            [
                "collect-repr-stages",
                "--model",
                "OpenVLA",
                "--dataset",
                "libero_10",
                "--manifest",
                str(manifest),
                "--hidden-state-root",
                str(hidden_root),
                "--bank-root",
                str(bank_root),
                "--token-group",
                "vision_pooled=0:2",
                "--token-group",
                "text_pooled=2:6",
                "--output",
                str(report),
            ]
        )
        == 0
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert len(payload["results"]) == 3
    assert (bank_root / "C0_bank.npz").exists()
    assert (bank_root / "C1_bank.npz").exists()
    assert (bank_root / "C2_bank.npz").exists()
