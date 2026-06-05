from __future__ import annotations

import json
from pathlib import Path

from vla_trace.cli import main
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
