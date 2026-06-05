from __future__ import annotations

import json

import numpy as np
import pytest

from vla_trace.behavior import (
    ImageMaskEvalConfig,
    apply_image_mask_to_obs_inplace,
    apply_patch_mask,
    apply_patch_mask_to_views,
    build_input_edit_record,
    build_patch_mask_from_maps,
    compute_attention_metrics,
    detect_instance_seg_keys,
)
from vla_trace.cli import main


def test_attention_metrics_mass_iou_and_hit():
    attention = np.zeros((4, 4), dtype=np.float64)
    attention[1, 1] = 10.0
    attention[0, 0] = 1.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True

    rows = compute_attention_metrics(attention, {"target": mask}, grid_shape=(4, 4), top_percent=10.0)

    assert rows[0]["instance_name"] == "target"
    assert rows[0]["attn_mass"] == pytest.approx(10.0 / 11.0)
    assert rows[0]["peak_hit"] == 1.0
    assert rows[0]["iou_top10"] > 0.0


@pytest.mark.parametrize("shape", [(2, 16), (3, 16)])
def test_attention_metrics_accepts_action_or_head_by_patch_arrays(shape):
    attention = np.zeros(shape, dtype=np.float64)
    attention[:, 5] = np.arange(1, shape[0] + 1)
    mask = np.zeros((4, 4), dtype=bool)
    mask.reshape(-1)[5] = True

    rows = compute_attention_metrics(attention, {"target": mask}, grid_shape=(4, 4), top_percent=10.0)

    assert rows[0]["attn_mass"] == pytest.approx(1.0)
    assert rows[0]["peak_hit"] == 1.0


def test_cli_attention_metrics_from_npz_artifacts(tmp_path):
    attention_path = tmp_path / "attention_maps.npz"
    masks_path = tmp_path / "step_masks.npz"
    metadata_path = tmp_path / "metadata.json"
    objects_path = tmp_path / "step_objects.json"
    output_csv = tmp_path / "metrics.csv"
    summary_json = tmp_path / "summary.json"
    plot_summary_csv = tmp_path / "attention_plot.csv"

    attention = np.zeros((4, 4), dtype=np.float32)
    attention[2, 2] = 5.0
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 1
    np.savez(attention_path, step_000=attention)
    np.savez(masks_path, step_000_target_object=mask)
    metadata_path.write_text(
        json.dumps({"trace_id": "trace0", "model": "OpenVLA", "dataset": "libero_10", "task_id": 0, "split_step": 1}),
        encoding="utf-8",
    )
    objects_path.write_text(
        json.dumps([{"step": 0, "objects": [{"mask_key_suffix": "target_object", "category": "object"}]}]),
        encoding="utf-8",
    )

    assert main(
        [
            "attention-metrics",
            "--attention",
            str(attention_path),
            "--masks",
            str(masks_path),
            "--metadata",
            str(metadata_path),
            "--objects",
            str(objects_path),
            "--output",
            str(output_csv),
            "--summary",
            str(summary_json),
            "--plot-summary",
            str(plot_summary_csv),
            "--grid-size",
            "4",
        ]
    ) == 0

    text = output_csv.read_text(encoding="utf-8")
    assert "target_object" in text
    assert "object" in text
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["n_rows"] == 1
    plot_summary_text = plot_summary_csv.read_text(encoding="utf-8")
    assert "mean_iou" in plot_summary_text
    assert "iou_top10" in plot_summary_text


def test_cli_attention_metrics_plot_summary_feeds_plot_attention(tmp_path):
    pytest.importorskip("matplotlib")
    attention_path = tmp_path / "attention_maps.npz"
    masks_path = tmp_path / "step_masks.npz"
    metadata_path = tmp_path / "metadata.json"
    metrics_csv = tmp_path / "metrics.csv"
    plot_summary_csv = tmp_path / "attention_plot.csv"
    figure = tmp_path / "attention_plot.png"

    attention = np.zeros((2, 16), dtype=np.float32)
    attention[:, 6] = [1.0, 3.0]
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask.reshape(-1)[6] = 1
    np.savez(attention_path, step_000=attention)
    np.savez(masks_path, step_000_target=mask)
    metadata_path.write_text(
        json.dumps({"trace_id": "trace0", "model": "pi0.5", "dataset": "libero_goal", "task_id": 3, "split_step": 1}),
        encoding="utf-8",
    )

    assert main(
        [
            "attention-metrics",
            "--attention",
            str(attention_path),
            "--masks",
            str(masks_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(metrics_csv),
            "--plot-summary",
            str(plot_summary_csv),
            "--grid-size",
            "4",
        ]
    ) == 0
    assert main(["plot-attention", str(plot_summary_csv), "--metric", "iou_top10", "--output", str(figure)]) == 0
    assert figure.exists()


def test_cli_attention_export_qualitative_views(tmp_path):
    raw = np.zeros((2, 1, 6, 6), dtype=np.float32)
    raw[..., :] = 1.0
    raw[:, :, 5, 0:2] = 4.0
    raw_path = tmp_path / "raw_attention.npz"
    output = tmp_path / "views.npz"
    np.savez(raw_path, step_000=raw)

    assert main(
        [
            "attention-export",
            "--attention",
            str(raw_path),
            "--key",
            "step_000",
            "--visual-span",
            "0:2",
            "--text-span",
            "2:5",
            "--action-span",
            "5:6",
            "--output",
            str(output),
        ]
    ) == 0

    with np.load(output, allow_pickle=False) as payload:
        assert payload["action_to_image"].shape == (2,)
        assert payload["action_to_text"].shape == (3,)
        assert payload["text_to_image"].shape == (3, 2)
        assert payload["layer_modality_mass"].shape == (2, 3)
        assert payload["layer_modality_flow"].shape == (2, 3, 3)


def test_patchmask_black_and_background_selection():
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    image[0, 0] = [200, 200, 200]
    target = np.zeros((4, 4), dtype=bool)
    target[1:3, 1:3] = True
    robot = np.zeros((4, 4), dtype=bool)
    robot[0, :] = True

    selected = build_patch_mask_from_maps(
        {"target": target, "robot_arm": robot},
        variant="mask_target",
        categories={"target": "object", "robot_arm": "robot_arm"},
    )
    masked = apply_patch_mask(image, selected, mode="black", mask_value=0)

    assert selected.sum() == 4
    assert masked[1, 1].tolist() == [0, 0, 0]
    background = build_patch_mask_from_maps({"target": target, "robot_arm": robot}, variant="mask_background")
    assert background[3, 3]


def test_patchmask_libero_obs_inplace_multiview():
    class Model:
        instances_to_ids = {"Panda0": 0, "PandaGripper0": 1, "moka_pot_1": 2}

    class InnerEnv:
        model = Model()

    class Env:
        env = InnerEnv()
        obj_of_interest = ["moka_pot_1"]

        def get_segmentation_instances(self, seg):
            plane = seg[..., 0] if seg.ndim == 3 else seg
            return {
                "robot": (plane == 1).astype(np.uint8),
                "gripper": (plane == 2).astype(np.uint8),
                "moka_pot_1": (plane == 3).astype(np.uint8),
            }

    seg_agent = np.zeros((4, 4, 1), dtype=np.int32)
    seg_agent[1:3, 1:3, 0] = 3
    seg_wrist = np.zeros((4, 4, 1), dtype=np.int32)
    seg_wrist[0:2, 0:2, 0] = 3
    obs = {
        "agentview_image": np.full((4, 4, 3), 100, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((4, 4, 3), 80, dtype=np.uint8),
        "agentview_instance_segmentation": seg_agent,
        "robot0_eye_in_hand_instance_segmentation": seg_wrist,
    }

    assert detect_instance_seg_keys(obs) == (
        "agentview_instance_segmentation",
        "robot0_eye_in_hand_instance_segmentation",
    )
    apply_image_mask_to_obs_inplace(obs, Env(), ImageMaskEvalConfig(variant="mask_target", mode="black"))

    assert obs["agentview_image"][1, 1].tolist() == [0, 0, 0]
    assert obs["robot0_eye_in_hand_image"][0, 0].tolist() == [0, 0, 0]


def test_patchmask_offline_multi_view():
    image_a = np.full((3, 3, 3), 20, dtype=np.uint8)
    image_b = np.full((3, 3, 3), 30, dtype=np.uint8)
    mask_a = np.zeros((3, 3), dtype=np.uint8)
    mask_b = np.zeros((3, 3), dtype=np.uint8)
    mask_a[0, 0] = 1
    mask_b[1, 1] = 1

    outputs = apply_patch_mask_to_views(
        {"agent": image_a, "wrist": image_b},
        {"agent": {"target": mask_a}, "wrist": {"target": mask_b}},
        variant="custom",
        mode="black",
        instances=["target"],
    )

    assert outputs["agent"][0, 0].tolist() == [0, 0, 0]
    assert outputs["wrist"][1, 1].tolist() == [0, 0, 0]


def test_cli_patchmask_writes_masked_image_and_manifest(tmp_path):
    image = np.full((4, 4, 3), 50, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    image_path = tmp_path / "image.npy"
    masks_path = tmp_path / "masks.npz"
    out_image = tmp_path / "masked.npy"
    manifest = tmp_path / "manifest.json"
    np.save(image_path, image)
    np.savez(masks_path, target=mask)

    assert main(
        [
            "patchmask",
            "--image",
            str(image_path),
            "--masks",
            str(masks_path),
            "--variant",
            "custom",
            "--mode",
            "black",
            "--instance",
            "target",
            "--output-image",
            str(out_image),
            "--output-manifest",
            str(manifest),
        ]
    ) == 0

    masked = np.load(out_image, allow_pickle=False)
    assert masked[0, 0].tolist() == [0, 0, 0]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["variant"] == "custom"
    assert payload["selected_pixels"] == 1


def test_input_edit_manifest_and_summary(tmp_path):
    record = build_input_edit_record(
        edit_id="edit_moka_to_bowl",
        task_id=2,
        edit_type="instruction_replace",
        base_instruction="put the moka pot on the stove",
        edited_instruction="put the bowl on the stove",
        target_object="moka pot",
        replacement_object="bowl",
        expected_shift="attention and action should follow bowl",
    )
    assert record["edited_instruction"].startswith("put the bowl")

    manifest = tmp_path / "edits.jsonl"
    assert main(
        [
            "input-edit",
            "--output",
            str(manifest),
            "--edit-id",
            "edit0",
            "--task-id",
            "0",
            "--edit-type",
            "instruction_replace",
            "--base-instruction",
            "pick up the cup",
            "--edited-instruction",
            "pick up the plate",
            "--target-object",
            "cup",
            "--replacement-object",
            "plate",
        ]
    ) == 0
    assert "pick up the plate" in manifest.read_text(encoding="utf-8")

    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps({"edit_id": "edit0", "edit_type": "instruction_replace", "success": True}),
                json.dumps({"edit_id": "edit1", "edit_type": "instruction_replace", "success": False}),
            ]
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    assert main(["input-edit", "--results", str(results), "--output", str(summary)]) == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["groups"][0]["success_rate"] == 0.5


def test_cli_attention_overlay(tmp_path):
    pytest.importorskip("matplotlib")
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    attention = np.zeros((2, 16), dtype=np.float32)
    attention[:, 5] = [1.0, 3.0]
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask.reshape(-1)[5] = 1
    image_path = tmp_path / "image.npy"
    attention_path = tmp_path / "attention.npz"
    mask_path = tmp_path / "mask.npz"
    output = tmp_path / "overlay.png"
    np.save(image_path, image)
    np.savez(attention_path, step_000=attention)
    np.savez(mask_path, step_000_target=mask)

    assert main(
        [
            "attention-overlay",
            "--image",
            str(image_path),
            "--attention",
            str(attention_path),
            "--attention-key",
            "step_000",
            "--masks",
            str(mask_path),
            "--mask-key",
            "step_000_target",
            "--output",
            str(output),
            "--grid-size",
            "4",
        ]
    ) == 0
    assert output.exists()
