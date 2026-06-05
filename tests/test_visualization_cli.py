from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from vla_trace.cli import main
from vla_trace.visualization.cka_publication import _image_text_series
from vla_trace.visualization.knockout import collect_knockout_results


def test_cli_plot_cka_cross_modal_report(tmp_path, capsys):
    report = tmp_path / "cross_modal_cka_report.json"
    figure = tmp_path / "cka.png"
    source = tmp_path / "cka.csv"
    report.write_text(
        json.dumps(
            {
                "status": "ok",
                "analysis": "cross_modal",
                "layerwise": False,
                "profiles": {
                    "C0": {
                        "vision_pooled": {"vision_pooled": 1.0, "text_pooled": 0.52},
                        "text_pooled": {"vision_pooled": 0.52, "text_pooled": 1.0},
                    },
                    "C1": {
                        "vision_pooled": {"vision_pooled": 1.0, "text_pooled": 0.61},
                        "text_pooled": {"vision_pooled": 0.61, "text_pooled": 1.0},
                    },
                },
                "checkpoints": ["C0", "C1"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["plot-cka", str(report), "--output", str(figure), "--source-data", str(source)]) == 0

    assert str(figure) in capsys.readouterr().out
    assert figure.exists()
    assert "vision_pooled" in source.read_text(encoding="utf-8")


def test_cli_plot_cka_layerwise_drift_report(tmp_path):
    report = tmp_path / "checkpoint_drift_cka_report.json"
    figure = tmp_path / "drift.png"
    report.write_text(
        json.dumps(
            {
                "status": "ok",
                "analysis": "checkpoint_drift",
                "layerwise": True,
                "view": "joint_pooled",
                "reference": "C0",
                "profiles": {
                    "C0": {"0": 1.0, "1": 1.0},
                    "C1": {"0": 0.92, "1": 0.84},
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["plot-cka", str(report), "--output", str(figure)]) == 0
    assert figure.exists()


def test_cli_plot_cka_legacy_flat_layer_profiles(tmp_path):
    report = tmp_path / "legacy_cross_modal_cka_report.json"
    figure = tmp_path / "legacy_cka.png"
    source = tmp_path / "legacy_cka.csv"
    report.write_text(
        json.dumps(
            {
                "status": "ok",
                "arch": "openvla",
                "analysis": "cross_modal_cka",
                "profiles": {
                    "C0": {"0": 0.81, "1": 0.87},
                    "C1": {"0": 0.72, "1": 0.80},
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["plot-cka", str(report), "--output", str(figure), "--source-data", str(source)]) == 0

    assert figure.exists()
    source_text = source.read_text(encoding="utf-8")
    assert "checkpoint,layer,cka" in source_text
    assert "C0,0,0.81" in source_text


def test_cli_plot_knockout_results_with_explicit_schema(tmp_path):
    root = tmp_path / "knockout"
    root.mkdir()
    for layer, success in [(0, 8), (1, 6)]:
        (root / f"layer_{layer}.json").write_text(
            json.dumps(
                {
                    "model": "OpenVLA",
                    "dataset": "libero_10",
                    "setting": "generation_no_image",
                    "layer": layer,
                    "window": 7,
                    "success_num": success,
                    "test_num": 10,
                }
            ),
            encoding="utf-8",
        )
    (root / "baseline.json").write_text(
        json.dumps(
            {
                "model": "OpenVLA",
                "dataset": "libero_10",
                "setting": "baseline",
                "success_rate": 0.9,
            }
        ),
        encoding="utf-8",
    )
    figure = tmp_path / "knockout.png"
    source = tmp_path / "knockout.csv"

    assert main(
        [
            "plot-knockout",
            str(root),
            "--output",
            str(figure),
            "--model",
            "OpenVLA",
            "--dataset",
            "libero_10",
            "--source-data",
            str(source),
        ]
    ) == 0

    assert figure.exists()
    assert "generation_no_image" in source.read_text(encoding="utf-8")


def test_cli_plot_knockout_line_grid_from_publication_csv(tmp_path):
    selected = tmp_path / "main_layerwise_selected.csv"
    baseline = tmp_path / "baselines.csv"
    output_dir = tmp_path / "line_figures"
    source_dir = tmp_path / "line_source"
    selected.write_text(
        "\n".join(
            [
                "model,dataset,main_setting_label,figure_protocol,layer,window,success_rate,ci_low,ci_high,canonical_setting",
                "openvla,libero_10,Generation: no image,window7,0,7,80,72,88,generation_no_image",
                "openvla,libero_10,Generation: no image,window7,8,7,62,54,70,generation_no_image",
                "openvla,libero_10,Generation: no text,window7,0,7,76,68,84,generation_no_text",
                "openvla,libero_10,Generation: no text,window7,8,7,70,62,78,generation_no_text",
            ]
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        "\n".join(
            [
                "model,dataset,baseline_success_rate",
                "openvla,libero_10,90",
            ]
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "plot-knockout-line",
            "--selected-csv",
            str(selected),
            "--baseline-csv",
            str(baseline),
            "--output-dir",
            str(output_dir),
            "--source-data-dir",
            str(source_dir),
            "--model",
            "OpenVLA",
            "--dataset",
            "libero_10",
        ]
    ) == 0

    assert (output_dir / "fig_line_layerwise_openvla.png").exists()
    assert (output_dir / "fig_line_layerwise_openvla.pdf").exists()
    assert (output_dir / "fig_line_layerwise_openvla.svg").exists()
    exported = (source_dir / "main_line_layerwise_selected.csv").read_text(encoding="utf-8")
    assert "Generation: no image" in exported
    assert "success_rate_fraction" in exported or "canonical_setting" in exported


def test_knockout_parser_keeps_legacy_path_inference(tmp_path):
    result_dir = tmp_path / "openvla_layerwise_prefill_no_image__generation_no_text_window7" / "layer31_w7-centercrop"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "libero_spatial_0.7550.json"
    result_path.write_text(
        json.dumps({"success_num": 15, "test_num": 20}),
        encoding="utf-8",
    )

    rows = collect_knockout_results([tmp_path])

    assert len(rows) == 1
    assert rows[0].model == "openvla"
    assert rows[0].dataset == "libero_spatial"
    assert rows[0].layer == 31
    assert rows[0].window == 7


def test_knockout_parser_handles_oft_legacy_paths(tmp_path):
    result_dir = (
        tmp_path
        / "eval"
        / "logs"
        / "oft"
        / "oft_knockout_layerwise_release"
        / "libero_object"
        / "layerwise"
        / "no_text_full"
        / "layer9_w7"
    )
    result_dir.mkdir(parents=True)
    result_path = result_dir / "libero_object_0.0000.json"
    result_path.write_text(json.dumps({"success_num": 0, "test_num": 20}), encoding="utf-8")

    rows = collect_knockout_results([tmp_path])

    assert len(rows) == 1
    assert rows[0].model == "openvla_oft"
    assert rows[0].dataset == "libero_object"
    assert rows[0].setting == "openvla_oft_layerwise_generation_no_text_full_window7"
    assert rows[0].canonical_setting == "generation_no_text_full"
    assert rows[0].layer == 9
    assert rows[0].window == 7


def test_plot_knockout_line_from_raw_json_exports_public_schema_columns(tmp_path):
    root = tmp_path / "openvla" / "libero_10" / "trials20" / "openvla_layerwise_generation_no_image_window7"
    for layer, success in [(0, 10), (8, 5)]:
        result_dir = root / f"layer{layer}_w7-centercrop"
        result_dir.mkdir(parents=True)
        (result_dir / f"libero_10_{success / 20:.4f}.json").write_text(
            json.dumps({"success_num": success, "test_num": 20}),
            encoding="utf-8",
        )
    out_dir = tmp_path / "figures"
    source_dir = tmp_path / "source"

    assert main(
        [
            "plot-knockout-line",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--source-data-dir",
            str(source_dir),
            "--model",
            "OpenVLA",
            "--dataset",
            "libero_10",
        ]
    ) == 0

    header = (source_dir / "main_line_layerwise_selected.csv").read_text(encoding="utf-8").splitlines()[0]
    for column in (
        "source_group",
        "protocol",
        "crop",
        "success_rate_fraction",
        "is_layerwise",
        "is_baseline",
        "is_all_layers",
        "intervention_scope",
    ):
        assert column in header


def test_cli_plot_attention_iou_csv(tmp_path):
    csv_path = tmp_path / "attention_iou.csv"
    figure = tmp_path / "attention_iou.png"
    csv_path.write_text(
        "\n".join(
            [
                "task_id,phase,phase_label,metric,metric_label,mean_iou,std_iou,n_steps",
                "0,phase1,Phase 1,iou_top10,Top-10 IoU,0.32,0.03,8",
                "0,phase2,Phase 2,iou_top10,Top-10 IoU,0.46,0.04,8",
                "1,phase1,Phase 1,iou_top10,Top-10 IoU,0.28,0.02,7",
                "1,phase2,Phase 2,iou_top10,Top-10 IoU,0.41,0.05,7",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["plot-attention", str(csv_path), "--output", str(figure), "--metric", "iou_top10"]) == 0
    assert figure.exists()


def test_cli_plot_attention_accepts_publication_source_csv(tmp_path):
    csv_path = tmp_path / "publication_attention_iou.csv"
    figure = tmp_path / "publication_attention_iou.png"
    csv_path.write_text(
        "\n".join(
            [
                "task_id,instruction,success,total_steps,phase,phase_label,metric,metric_label,mean_iou,std_iou,n_steps",
                "0,put both objects in the basket,yes,120,phase1,Phase 1,iou_top10_gt,Top-10 patch IoU,0.32,0.03,8",
                "0,put both objects in the basket,yes,120,phase2,Phase 2,iou_top10_gt,Top-10 patch IoU,0.46,0.04,8",
                "0,put both objects in the basket,yes,120,phase1,Phase 1,iou_fixed_gt,Fixed-threshold IoU,0.22,0.02,8",
                "0,put both objects in the basket,yes,120,phase2,Phase 2,iou_fixed_gt,Fixed-threshold IoU,0.31,0.03,8",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["plot-attention", str(csv_path), "--output", str(figure), "--metric", "iou_top10_gt"]) == 0
    assert figure.exists()


def test_cli_plot_cka_publication_from_explicit_report_map(tmp_path):
    align = tmp_path / "openvla_alignment.json"
    drift = tmp_path / "openvla_drift.json"
    output_dir = tmp_path / "cka_publication"
    align.write_text(
        json.dumps(
            {
                "analysis": "cross_modal_cka",
                "profiles": {
                    "C0": {"1": 0.35, "8": 0.42},
                    "C1": {"1": 0.55, "8": 0.63},
                    "C2": {"1": 0.61, "8": 0.71},
                },
            }
        ),
        encoding="utf-8",
    )
    drift.write_text(
        json.dumps(
            {
                "analysis": "checkpoint_drift",
                "matched_layer_summary": {
                    "targets": {
                        "C1": {
                            "vision_pooled": {
                                "status": "ok",
                                "per_layer": {"1": 0.85, "8": 0.75},
                            }
                        },
                        "C2": {
                            "vision_pooled": {
                                "status": "ok",
                                "per_layer": {"1": 0.72, "8": 0.66},
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "plot-cka-publication",
            "--report",
            f"openvla:libero_10:alignment={align}",
            "--report",
            f"openvla:libero_10:drift={drift}",
            "--output-dir",
            str(output_dir),
            "--datasets",
            "libero_10",
            "--models",
            "OpenVLA",
        ]
    ) == 0

    assert (output_dir / "image_text_cka_panel.png").exists()
    assert (output_dir / "drift_cka_panel_vision_pooled.png").exists()
    assert (output_dir / "cka_publication_main.png").exists()


def test_cli_plot_cka_publication_accepts_zero_based_layer_profiles(tmp_path):
    align = tmp_path / "openvla_alignment_zero_based.json"
    output_dir = tmp_path / "cka_publication_zero_based"
    align.write_text(
        json.dumps(
            {
                "analysis": "cross_modal_cka",
                "profiles": {
                    "C0": {"0": 0.35},
                    "C1": {"0": 0.55},
                    "C2": {"0": 0.61},
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "plot-cka-publication",
            "--report",
            f"openvla:libero_10:alignment={align}",
            "--output-dir",
            str(output_dir),
            "--datasets",
            "libero_10",
            "--models",
            "OpenVLA",
        ]
    ) == 0

    assert (output_dir / "image_text_cka_panel.png").exists()


def test_cka_publication_zero_based_profiles_are_not_dropped() -> None:
    series = _image_text_series({"profiles": {"C0": {"0": 0.81}}}, "C0")

    assert series is not None
    layers, mean, low, high = series
    assert layers.tolist() == [1]
    assert mean.tolist() == [0.81]
    assert low.tolist() == [0.81]
    assert high.tolist() == [0.81]


def test_cli_plot_cka_publication_supports_coco_panel(tmp_path):
    align = tmp_path / "pi05_coco_alignment.json"
    output_dir = tmp_path / "cka_publication_coco"
    align.write_text(
        json.dumps(
            {
                "metrics": {
                    "cka": {
                        "baseline": {
                            "mean": {"C0": {"1": 0.42}, "C1": {"1": 0.51}, "C2": {"1": 0.55}},
                            "ci_low": {"C0": {"1": 0.40}},
                            "ci_high": {"C0": {"1": 0.44}},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "plot-cka-publication",
            "--report",
            f"pi05:coco:alignment={align}",
            "--output-dir",
            str(output_dir),
            "--datasets",
            "coco",
            "--models",
            "pi0.5",
        ]
    ) == 0

    assert (output_dir / "image_text_cka_panel.svg").exists()


def test_attention_iou_csv_requires_value_column(tmp_path):
    csv_path = tmp_path / "bad_attention_iou.csv"
    figure = tmp_path / "bad_attention_iou.png"
    csv_path.write_text(
        "\n".join(
            [
                "task_id,phase,metric,n_steps",
                "0,phase1,iou_top10,8",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mean_iou"):
        main(["plot-attention", str(csv_path), "--output", str(figure)])


def test_cli_plot_attention_map_bar_and_heatmap(tmp_path):
    vector_path = tmp_path / "action_to_text.npz"
    heatmap_path = tmp_path / "token_to_image.npy"
    bar = tmp_path / "action_to_text.png"
    heatmap = tmp_path / "token_to_image.png"
    np.savez(vector_path, step_030=np.asarray([0.2, 0.5, 0.3], dtype=np.float32))
    np.save(heatmap_path, np.asarray([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], dtype=np.float32))

    assert main(
        [
            "plot-attention-map",
            str(vector_path),
            "--key",
            "step_030",
            "--keep-last-dims",
            "1",
            "--kind",
            "bar",
            "--normalize",
            "sum",
            "--x-labels",
            "bos,instruction,newline",
            "--output",
            str(bar),
        ]
    ) == 0
    assert main(
        [
            "plot-attention-map",
            str(heatmap_path),
            "--kind",
            "heatmap",
            "--normalize",
            "row",
            "--x-labels",
            "patch0,patch1,patch2",
            "--y-labels",
            "token0,token1",
            "--output",
            str(heatmap),
        ]
    ) == 0

    assert bar.exists()
    assert heatmap.exists()
