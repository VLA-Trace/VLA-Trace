from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vla_trace.cli import main
from vla_trace.representations.bank import save_representation_bank


ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_runs(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "vla-trace" in out
    assert "inspect-model" in out


def test_inspect_model_openvla(capsys):
    code = main(["inspect-model", str(ROOT / "configs/models/openvla.yaml")])
    assert code == 0
    out = capsys.readouterr().out
    assert "family: openvla" in out
    assert "visual_tokens: 256" in out
    assert "supports_stage3: True" in out


def test_inspect_model_relative_config_from_other_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = main(["inspect-model", "configs/models/pi05.yaml"])

    assert code == 0
    out = capsys.readouterr().out
    assert "family: pi05" in out
    assert "supported_stages: stage1, stage2, stage3" in out


def test_doctor_resolves_default_configs_from_other_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["doctor", "--model", "OpenVLA", "--dataset", "libero_10"]) == 0

    payload = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["model_config"]["status"] == "ok"
    assert checks["benchmark_config"]["status"] == "ok"
    assert checks["model_path"]["status"] == "warning"
    assert checks["data_root"]["status"] == "warning"
    assert payload["summary"]["error"] == 0


def test_collect_repr_records_user_paths(tmp_path):
    output = tmp_path / "collect.json"

    assert main(
        [
            "collect-repr",
            str(ROOT / "configs/experiments/openvla_libero_repr.yaml"),
            "--model-path",
            "checkpoints/openvla",
            "--data-root",
            "datasets/LIBERO",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["model_path"] == "checkpoints/openvla"
    assert payload["config"]["data_root"] == "datasets/LIBERO"


def test_collect_repr_builds_bank_from_hidden_state_artifacts(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    bank = tmp_path / "bank.npz"
    report = tmp_path / "collect_report.json"
    rows = []
    for idx in range(2):
        sample_id = f"s{idx}"
        hidden = np.stack(
            [
                np.arange(6 * 4, dtype=np.float32).reshape(6, 4) + idx,
                np.ones((6, 4), dtype=np.float32) * (idx + 1),
            ],
            axis=0,
        )
        np.savez(hidden_dir / f"{sample_id}.npz", hidden_states=hidden)
        rows.append({"sample_id": sample_id, "image_path": f"images/{idx}.png", "instruction": "pick object"})
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert main(
        [
            "collect-repr",
            "--model",
            "OpenVLA",
            "--manifest",
            str(manifest),
            "--hidden-state-dir",
            str(hidden_dir),
            "--bank-output",
            str(bank),
            "--token-group",
            "vision_pooled=0:2",
            "--token-group",
            "text_pooled=2:6",
            "--output",
            str(report),
        ]
    ) == 0

    assert bank.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert "vision_pooled/layer_0" in payload["array_keys"]


def test_convert_bank_cli_round_trip(tmp_path):
    source = tmp_path / "bank.json"
    target = tmp_path / "bank.npz"
    save_representation_bank(source, {"joint_pooled": [[1.0, 0.0], [0.0, 1.0]]}, metadata={"stage": "C1"})

    assert main(["convert-bank", "--input", str(source), "--output", str(target), "--metadata", "model=OpenVLA"]) == 0

    assert target.exists()


def test_doctor_checks_local_artifacts(tmp_path):
    model_dir = tmp_path / "checkpoint"
    data_dir = tmp_path / "libero"
    model_dir.mkdir()
    data_dir.mkdir()
    bank_path = tmp_path / "bank.json"
    attention_path = tmp_path / "attention.npz"
    masks_path = tmp_path / "masks.npz"
    results_path = tmp_path / "result.json"
    input_edits_path = tmp_path / "edits.jsonl"
    report_path = tmp_path / "doctor.json"

    save_representation_bank(bank_path, {"joint_pooled": [[1.0, 0.0], [0.0, 1.0]]})
    np.savez(attention_path, step_000=np.ones((4, 4), dtype=np.float32))
    np.savez(masks_path, step_000_target=np.ones((4, 4), dtype=np.uint8))
    results_path.write_text(
        json.dumps(
            {
                "model": "OpenVLA",
                "dataset": "libero_10",
                "setting": "generation_no_image",
                "success_num": 1,
                "test_num": 1,
            }
        ),
        encoding="utf-8",
    )
    input_edits_path.write_text(
        json.dumps(
            {
                "edit_id": "edit0",
                "edit_type": "instruction_replace",
                "base_instruction": "pick up the cup",
                "edited_instruction": "pick up the bowl",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "doctor",
            "--model",
            "OpenVLA",
            "--dataset",
            "libero_10",
            "--model-path",
            str(model_dir),
            "--data-root",
            str(data_dir),
            "--bank",
            f"C1={bank_path}",
            "--attention",
            str(attention_path),
            "--masks",
            str(masks_path),
            "--results",
            str(results_path),
            "--input-edits",
            str(input_edits_path),
            "--output",
            str(report_path),
            "--strict",
        ]
    ) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["summary"]["error"] == 0
