from __future__ import annotations

import math
import json

from vla_trace.knockout import (
    KnockoutSpec,
    build_additive_mask,
    build_sweep_from_config,
    expand_layer_window,
    make_openvla_partitions,
    make_pi05_partitions,
)
from vla_trace.cli import main


def test_openvla_generation_no_image_blocks_action_to_visual() -> None:
    partitions = make_openvla_partitions(visual_tokens=4, text_tokens=3, action_tokens=2)
    spec = KnockoutSpec(family="openvla", phase="generation", mode="no_image", layers=(1,))
    mask = build_additive_mask(spec, partitions)

    assert math.isinf(mask.at(1, partitions.action.start, partitions.visual.start))
    assert mask.at(1, partitions.action.start, partitions.text.start) == 0.0


def test_openvla_prefill_no_image_preserves_image_self_attention_and_blocks_text_to_image() -> None:
    partitions = make_openvla_partitions(visual_tokens=4, text_tokens=3, action_tokens=2)
    spec = KnockoutSpec(family="openvla", phase="prefill", mode="no_image", layers=(0,))
    mask = build_additive_mask(spec, partitions)

    assert mask.at(0, partitions.visual.start, partitions.visual.start) == 0.0
    assert math.isinf(mask.at(0, partitions.text.start, partitions.visual.start))
    assert mask.at(0, partitions.text.start, partitions.text.start) == 0.0


def test_pi05_prefill_modes_block_bidirectional_visual_language_attention() -> None:
    partitions = make_pi05_partitions(visual_tokens=4, text_tokens=3, action_tokens=2)
    for mode in ("no_text", "no_image", "no_vl"):
        spec = KnockoutSpec(family="pi05", phase="prefill", mode=mode, layers=(0,))
        mask = build_additive_mask(spec, partitions)
        assert math.isinf(mask.at(0, partitions.visual.start, partitions.text.start))
        assert math.isinf(mask.at(0, partitions.text.start, partitions.visual.start))


def test_pi05_generation_no_text_blocks_action_to_text() -> None:
    partitions = make_pi05_partitions(visual_tokens=4, text_tokens=3, action_tokens=2)
    spec = KnockoutSpec(family="pi05", phase="generation", mode="no_text", layers=(2,))
    mask = build_additive_mask(spec, partitions)

    assert math.isinf(mask.at(2, partitions.action.start, partitions.text.start))
    assert mask.at(2, partitions.action.start, partitions.visual.start) == 0.0


def test_openvla_text_scope_bos_newline_blocks_structural_tokens_only() -> None:
    partitions = make_openvla_partitions(visual_tokens=2, text_tokens=3, action_tokens=1)
    spec = KnockoutSpec(
        family="openvla",
        phase="generation",
        mode="no_text",
        text_scope="bos_newline",
        layers=(0,),
    )
    mask = build_additive_mask(spec, partitions)

    newline_index = partitions.text.stop - 1
    instruction_index = partitions.text.start
    assert math.isinf(mask.at(0, partitions.action.start, partitions.prefix.start))
    assert math.isinf(mask.at(0, partitions.action.start, newline_index))
    assert mask.at(0, partitions.action.start, instruction_index) == 0.0


def test_pi05_text_scope_newline_only_preserves_instruction_tokens() -> None:
    partitions = make_pi05_partitions(visual_tokens=2, text_tokens=3, action_tokens=1)
    spec = KnockoutSpec(
        family="pi05",
        phase="generation",
        mode="no_text",
        text_scope="newline_only",
        layers=(0,),
    )
    mask = build_additive_mask(spec, partitions)

    newline_index = partitions.text.stop - 1
    instruction_index = partitions.text.start
    assert math.isinf(mask.at(0, partitions.action.start, newline_index))
    assert mask.at(0, partitions.action.start, instruction_index) == 0.0


def test_no_fusion_is_prefill_only() -> None:
    pi05 = make_pi05_partitions(visual_tokens=4, text_tokens=3, action_tokens=2)
    pi05_spec = KnockoutSpec(family="pi05", phase="generation", mode="no_fusion", layers=(0,))
    pi05_mask = build_additive_mask(pi05_spec, pi05)
    assert pi05_mask.at(0, pi05.action.start, pi05.visual.start) == 0.0
    assert pi05_mask.at(0, pi05.action.start, pi05.text.start) == 0.0

    openvla = make_openvla_partitions(visual_tokens=4, text_tokens=3, action_tokens=2)
    openvla_spec = KnockoutSpec(family="openvla", phase="generation", mode="no_fusion", layers=(0,))
    openvla_mask = build_additive_mask(openvla_spec, openvla)
    assert openvla_mask.at(0, openvla.action.start, openvla.visual.start) == 0.0
    assert openvla_mask.at(0, openvla.action.start, openvla.text.start) == 0.0


def test_window_layer_expansion() -> None:
    assert expand_layer_window(num_layers=8, center_layers=(0, 4, 7), window_size=3) == (0, 1, 3, 4, 5, 6, 7)

    sweep = build_sweep_from_config(
        {
            "family": "pi05",
            "phase": "generation",
            "mode": "no_text",
            "text_scope": "instruction",
            "num_layers": 18,
            "layers": {
                "type": "window",
                "center_layers": [0, 4, 8, 12, 17],
                "window_size": 5,
            },
            "token_layout": {
                "visual_tokens": 768,
                "text_tokens": 32,
                "action_tokens": 10,
            },
        }
    )

    assert sweep.spec.layers == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)


def test_cli_knockout_mask_artifact_is_strict_json(tmp_path) -> None:
    cfg = tmp_path / "knockout.yaml"
    out = tmp_path / "mask.json"
    cfg.write_text(
        """
family: openvla
phase: generation
mode: no_image
num_layers: 2
layers:
  type: explicit
  values: [0]
token_layout:
  visual_tokens: 2
  text_tokens: 2
  action_tokens: 1
""",
        encoding="utf-8",
    )

    assert main(["knockout", str(cfg), "--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Infinity" not in text
    payload = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert payload["spec"]["additive_block_value"] == "-inf"
    assert payload["mask"]["block_value"] == -1.0e9
    assert payload["mask"]["values"][0][payload["mask"]["shape"][1] - 1][1] == -1.0e9


def test_cli_pi05_knockout_mask_artifact_is_strict_json(tmp_path) -> None:
    cfg = tmp_path / "pi05_knockout.yaml"
    out = tmp_path / "pi05_mask.json"
    cfg.write_text(
        """
family: pi05
phase: generation
mode: no_text
num_layers: 2
layers:
  type: explicit
  values: [1]
token_layout:
  visual_tokens: 2
  text_tokens: 2
  action_tokens: 1
""",
        encoding="utf-8",
    )

    assert main(["knockout", str(cfg), "--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    payload = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert payload["spec"]["additive_block_value"] == "-inf"
    assert payload["mask"]["block_value"] == -1.0e9
    assert payload["mask"]["layers"] == [1]
    assert payload["config"]["token_layout"]["token_order"] == "text,visual,action"
    assert payload["mask"]["values"][0][payload["mask"]["shape"][1] - 1][0] == -1.0e9


def test_pi05_token_order_can_match_paper_or_adapter_layout() -> None:
    paper = make_pi05_partitions(visual_tokens=2, text_tokens=3, action_tokens=1)
    exported = make_pi05_partitions(visual_tokens=2, text_tokens=3, action_tokens=1, token_order="visual,text,action")

    assert paper.text.start == 0
    assert paper.visual.start == 3
    assert exported.visual.start == 0
    assert exported.text.start == 2


def test_cli_knockout_accepts_model_dataset_and_setting_flags(tmp_path, capsys) -> None:
    out = tmp_path / "mask.json"

    assert main(
        [
            "knockout",
            "--model",
            "OpenVLA",
            "--dataset",
            "libero_spatial",
            "--phase",
            "prefill",
            "--mode",
            "no_vl",
            "--layers",
            "0,1",
            "--visual-tokens",
            "2",
            "--text-tokens",
            "2",
            "--action-tokens",
            "1",
            "--output",
            str(out),
        ]
    ) == 0

    summary = capsys.readouterr().out
    assert "model=openvla" in summary
    assert "dataset=libero_spatial" in summary
    text = out.read_text(encoding="utf-8")
    assert "Infinity" not in text
    payload = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert payload["config"]["model"] == "configs/models/openvla.yaml"
    assert payload["config"]["benchmark"] == "configs/benchmarks/libero_spatial.yaml"
    assert payload["spec"]["phase"] == "prefill"
    assert payload["spec"]["mode"] == "no_vl"
    assert payload["mask"]["layers"] == [0, 1]


def test_cli_knockout_records_user_paths_in_artifact(tmp_path) -> None:
    out = tmp_path / "mask.json"

    assert main(
        [
            "knockout",
            "--model",
            "pi0.5",
            "--dataset",
            "libero_goal",
            "--model-config",
            "local/configs/pi05.yaml",
            "--benchmark-config",
            "local/configs/libero_goal.yaml",
            "--model-path",
            "checkpoints/pi05",
            "--data-root",
            "datasets/LIBERO",
            "--layers",
            "0",
            "--visual-tokens",
            "2",
            "--text-tokens",
            "2",
            "--action-tokens",
            "1",
            "--output",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["model"] == "local/configs/pi05.yaml"
    assert payload["config"]["benchmark"] == "local/configs/libero_goal.yaml"
    assert payload["config"]["model_path"] == "checkpoints/pi05"
    assert payload["config"]["data_root"] == "datasets/LIBERO"


def test_cli_knockout_accepts_all_layers_and_combined_phase_masks(tmp_path) -> None:
    out = tmp_path / "combined.json"

    assert main(
        [
            "knockout",
            "--model",
            "OpenVLA",
            "--dataset",
            "libero_10",
            "--mode",
            "openvla_prefill_no_image__generation_no_text",
            "--phase",
            "both",
            "--layers",
            "all",
            "--num-layers",
            "3",
            "--visual-tokens",
            "2",
            "--text-tokens",
            "2",
            "--action-tokens",
            "1",
            "--output",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mask"]["layers"] == [0, 1, 2]
    assert set(payload["phase_masks"]) == {"prefill", "generation"}
    assert payload["phase_specs"]["prefill"]["mode"] == "no_image"
    assert payload["phase_specs"]["generation"]["mode"] == "no_text"


def test_directional_knockout_blocks_only_requested_route() -> None:
    partitions = make_openvla_partitions(visual_tokens=2, text_tokens=2, action_tokens=1)
    spec = KnockoutSpec(
        family="openvla",
        phase="generation",
        mode="no_image",
        direction="text->action",
        layers=(0,),
        text_scope="all",
    )
    mask = build_additive_mask(spec, partitions)

    assert math.isinf(mask.at(0, partitions.action.start, partitions.text.start))
    assert mask.at(0, partitions.action.start, partitions.visual.start) == 0.0


def test_knockout_sweep_paper_0512_manifest(tmp_path) -> None:
    out = tmp_path / "sweep.json"

    assert main(
        [
            "knockout-sweep",
            "--model",
            "pi0.5",
            "--dataset",
            "libero_goal",
            "--num-layers",
            "2",
            "--window-size",
            "3",
            "--output",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["preset"] == "paper_0512"
    assert payload["n_jobs"] == 19
    tags = [job["tag"] for job in payload["jobs"]]
    assert any("pi05_layerwise_prefill_no_vl_window3/layer0_w3" in tag for tag in tags)
    assert any("pi05_all_layers_prefill_no_vl__generation_no_text" in tag for tag in tags)
