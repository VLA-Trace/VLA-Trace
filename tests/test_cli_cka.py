from __future__ import annotations

import json
from pathlib import Path

import yaml

from vla_trace.cli import main
from vla_trace.representations.bank import save_representation_bank


def test_cli_cka_with_saved_bank(tmp_path, capsys):
    c0 = tmp_path / "c0.json"
    c1 = tmp_path / "c1.json"
    save_representation_bank(
        c0,
        {
            "vision_pooled": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "text_pooled": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "joint_pooled": [[1.0, 1.0], [0.0, 1.0], [1.0, 2.0]],
        },
    )
    save_representation_bank(
        c1,
        {
            "vision_pooled": [[1.0, 0.1], [0.0, 1.0], [1.0, 1.1]],
            "text_pooled": [[1.0, 0.2], [0.0, 1.0], [1.0, 1.2]],
            "joint_pooled": [[1.0, 1.1], [0.0, 1.0], [1.0, 2.1]],
        },
    )
    config = tmp_path / "cka.yaml"
    output_dir = tmp_path / "out"
    config.write_text(
        yaml.safe_dump(
            {
                "analysis": "checkpoint_drift",
                "view": "joint_pooled",
                "bank_paths": {"C0": str(c0), "C1": str(c1)},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    assert main(["cka", str(config)]) == 0
    out = capsys.readouterr().out
    assert "checkpoint_drift" in out
    assert (output_dir / "checkpoint_drift_cka_report.json").exists()


def test_cli_cka_accepts_model_dataset_and_bank_flags(tmp_path, capsys):
    c0 = tmp_path / "c0.json"
    c1 = tmp_path / "c1.json"
    save_representation_bank(
        c0,
        {
            "vision_pooled": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "text_pooled": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "joint_pooled": [[1.0, 1.0], [0.0, 1.0], [1.0, 2.0]],
        },
    )
    save_representation_bank(
        c1,
        {
            "vision_pooled": [[1.0, 0.1], [0.0, 1.0], [1.0, 1.1]],
            "text_pooled": [[1.0, 0.2], [0.0, 1.0], [1.0, 1.2]],
            "joint_pooled": [[1.0, 1.1], [0.0, 1.0], [1.0, 2.1]],
        },
    )
    output_dir = tmp_path / "cli_out"

    assert main(
        [
            "cka",
            "--model",
            "pi0.5",
            "--dataset",
            "libero_object",
            "--model-path",
            "checkpoints/pi05",
            "--data-root",
            "datasets/LIBERO",
            "--analysis",
            "checkpoint_drift",
            "--view",
            "joint_pooled",
            "--bank",
            f"C0={c0}",
            "--bank",
            f"C1={c1}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    out = capsys.readouterr().out
    assert "checkpoint_drift" in out
    report = json.loads((output_dir / "checkpoint_drift_cka_report.json").read_text(encoding="utf-8"))
    assert report["model"] == "configs/models/pi05.yaml"
    assert report["benchmark"] == "configs/benchmarks/libero_object.yaml"
    assert report["model_path"] == "checkpoints/pi05"
    assert report["data_root"] == "datasets/LIBERO"


def test_cli_cka_print_config_shows_model_and_dataset(capsys):
    assert main(["cka", "--model", "OpenVLA", "--dataset", "libero_goal", "--print-config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "configs/models/openvla.yaml"
    assert payload["benchmark"] == "configs/benchmarks/libero_goal.yaml"


def test_cli_cka_records_user_relative_paths_without_rewriting(capsys):
    assert main(
        [
            "cka",
            "--model",
            "OpenVLA",
            "--dataset",
            "libero_10",
            "--model-config",
            "local/configs/my_openvla.yaml",
            "--benchmark-config",
            "local/configs/my_libero.yaml",
            "--model-path",
            "checkpoints/openvla",
            "--data-root",
            "datasets/LIBERO",
            "--print-config",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "local/configs/my_openvla.yaml"
    assert payload["benchmark"] == "local/configs/my_libero.yaml"
    assert payload["model_path"] == "checkpoints/openvla"
    assert payload["data_root"] == "datasets/LIBERO"


def test_cli_layerwise_drift_writes_matched_layer_summary(tmp_path):
    c0 = tmp_path / "c0.json"
    c1 = tmp_path / "c1.json"
    save_representation_bank(
        c0,
        {
            "vision_pooled/layer_0": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "vision_pooled/layer_1": [[0.8, 0.1], [0.1, 0.9], [0.9, 1.0]],
            "text_pooled/layer_0": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "text_pooled/layer_1": [[0.8, 0.1], [0.1, 0.9], [0.9, 1.0]],
            "joint_pooled/layer_0": [[1.0, 1.0], [0.0, 1.0], [1.0, 2.0]],
            "joint_pooled/layer_1": [[0.9, 1.0], [0.1, 1.1], [1.0, 2.1]],
        },
    )
    save_representation_bank(
        c1,
        {
            "vision_pooled/layer_0": [[1.0, 0.1], [0.0, 1.0], [1.0, 1.1]],
            "vision_pooled/layer_1": [[0.8, 0.2], [0.1, 0.9], [0.9, 1.1]],
            "text_pooled/layer_0": [[1.0, 0.1], [0.0, 1.0], [1.0, 1.1]],
            "text_pooled/layer_1": [[0.8, 0.2], [0.1, 0.9], [0.9, 1.1]],
            "joint_pooled/layer_0": [[1.0, 1.1], [0.0, 1.0], [1.0, 2.1]],
            "joint_pooled/layer_1": [[0.9, 1.1], [0.1, 1.1], [1.0, 2.2]],
        },
    )
    output_dir = tmp_path / "drift_out"

    assert main(
        [
            "cka",
            "--analysis",
            "checkpoint_drift",
            "--layerwise",
            "--view",
            "joint_pooled",
            "--reference",
            "C0",
            "--summary-views",
            "vision_pooled,text_pooled,joint_pooled",
            "--bank",
            f"C0={c0}",
            "--bank",
            f"C1={c1}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    report = json.loads((output_dir / "checkpoint_drift_cka_report.json").read_text(encoding="utf-8"))
    summary = report["matched_layer_summary"]
    assert summary["targets"]["C1"]["vision_pooled"]["status"] == "ok"
    assert summary["targets"]["C1"]["vision_pooled"]["n_layers"] == 2
    assert summary["targets"]["C1"]["joint_pooled"]["mean_cka"] > 0.9
