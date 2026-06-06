from __future__ import annotations

import json
from pathlib import Path

from vla_trace.cli import main
from vla_trace.evaluation.adapters import _to_action_array
from vla_trace.visualization.knockout import collect_knockout_results


ROOT = Path(__file__).resolve().parents[1]


def test_eval_libero_print_plan_from_public_config(capsys):
    assert (
        main(
            [
                "eval-libero",
                str(ROOT / "configs/experiments/openvla_libero_eval.yaml"),
                "--dataset",
                "libero_goal",
                "--task-ids",
                "0,1",
                "--num-trials-per-task",
                "2",
                "--dry-run",
                "--print-plan",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "eval-libero"
    assert payload["model"] == "openvla"
    assert payload["dataset"] == "libero_goal"
    assert payload["test_num"] == 4
    assert len(payload["jobs"]) == 4


def test_eval_libero_plan_records_unnorm_key(capsys):
    assert (
        main(
            [
                "eval-libero",
                "--model",
                "OpenVLA",
                "--dataset",
                "libero_10",
                "--unnorm-key",
                "libero_10_no_noops",
                "--dry-run",
                "--print-plan",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["unnorm_key"] == "libero_10_no_noops"


def test_action_tuple_returns_first_payload():
    action = _to_action_array(([1.0, 2.0, 3.0], {"hidden_states": "ignored"}))

    assert action.tolist() == [1.0, 2.0, 3.0]


def test_eval_libero_mock_result_is_plot_compatible(tmp_path):
    output_dir = tmp_path / "eval"

    assert (
        main(
            [
                "eval-libero",
                "--model",
                "OpenVLA",
                "--dataset",
                "libero_10",
                "--task-ids",
                "0",
                "--num-trials-per-task",
                "2",
                "--mock-env",
                "--phase",
                "generation",
                "--mode",
                "no_image",
                "--layers",
                "3",
                "--center-layer",
                "3",
                "--window-size",
                "7",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    rows = collect_knockout_results([output_dir])
    assert len(rows) == 1
    assert rows[0].model == "openvla"
    assert rows[0].dataset == "libero_10"
    assert rows[0].canonical_setting == "generation_no_image"
    assert rows[0].layer == 3
    assert rows[0].window == 7


def test_eval_libero_runs_knockout_manifest_subset(tmp_path):
    manifest = tmp_path / "sweep.json"
    output_dir = tmp_path / "runs"
    summary = tmp_path / "summary.json"

    assert (
        main(
            [
                "knockout-sweep",
                "--model",
                "pi0.5",
                "--dataset",
                "libero_object",
                "--window-size",
                "3",
                "--num-layers",
                "2",
                "--output",
                str(manifest),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "eval-libero",
                "--model",
                "pi0.5",
                "--dataset",
                "libero_object",
                "--knockout-manifest",
                str(manifest),
                "--max-jobs",
                "2",
                "--task-ids",
                "0",
                "--num-trials-per-task",
                "1",
                "--mock-env",
                "--output-dir",
                str(output_dir),
                "--output",
                str(summary),
            ]
        )
        == 0
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["n_jobs"] == 2
    assert len(payload["results"]) == 2
    assert len(list(output_dir.rglob("*.json"))) == 2


def test_eval_libero_patchmask_single_setting_mock_is_plot_compatible(tmp_path):
    output_dir = tmp_path / "patchmask_eval"

    assert (
        main(
            [
                "eval-libero",
                "--model",
                "OpenVLA",
                "--dataset",
                "libero_10",
                "--task-ids",
                "0",
                "--num-trials-per-task",
                "2",
                "--mock-env",
                "--patchmask-variant",
                "mask_target",
                "--patchmask-mode",
                "black",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    result_files = list(output_dir.rglob("*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["setting"] == "patchmask_mask_target_black"
    assert payload["patchmask_config"]["variant"] == "mask_target"
    assert payload["patchmask_config"]["mode"] == "black"
    rows = collect_knockout_results([output_dir])
    assert len(rows) == 1
    assert rows[0].canonical_setting == "patchmask_mask_target_black"


def test_patchmask_sweep_manifest_and_eval_subset(tmp_path):
    manifest = tmp_path / "patchmask_sweep.json"
    output_dir = tmp_path / "patchmask_runs"
    summary = tmp_path / "summary.json"

    assert (
        main(
            [
                "patchmask-sweep",
                "--model",
                "all",
                "--dataset",
                "all",
                "--output",
                str(manifest),
            ]
        )
        == 0
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["preset"] == "standard_libero_patchmask"
    assert payload["n_jobs"] == 120
    assert payload["jobs"][0]["setting"] == "baseline"
    assert any(job["setting"] == "patchmask_mask_robot_exc_gripper_mosaic" for job in payload["jobs"])
    assert not any(
        job["variant"] == "mask_background" and job["mode"] == "background_fill" for job in payload["jobs"]
    )

    assert (
        main(
            [
                "eval-libero",
                "--patchmask-manifest",
                str(manifest),
                "--job-tag",
                "openvla/libero_10/patchmask_mask_target_black",
                "--task-ids",
                "0",
                "--num-trials-per-task",
                "1",
                "--mock-env",
                "--output-dir",
                str(output_dir),
                "--output",
                str(summary),
            ]
        )
        == 0
    )
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["n_jobs"] == 1
    result = summary_payload["results"][0]
    assert result["model"] == "openvla"
    assert result["dataset"] == "libero_10"
    assert result["setting"] == "patchmask_mask_target_black"
    assert result["patchmask_config"]["variant"] == "mask_target"


def test_eval_libero_patchmask_plan_requires_instance_segmentation(capsys):
    assert (
        main(
            [
                "eval-libero",
                "--model",
                "pi0.5",
                "--dataset",
                "libero_goal",
                "--task-ids",
                "0",
                "--num-trials-per-task",
                "1",
                "--patchmask-variant",
                "mask_gripper",
                "--patchmask-mode",
                "mosaic",
                "--dry-run",
                "--print-plan",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["requires_instance_segmentation"] is True
    assert payload["patchmask_config"] == {
        "variant": "mask_gripper",
        "mode": "mosaic",
        "mask_value": 0,
        "bg_ring_width": 8,
        "mosaic_block": 8,
    }
