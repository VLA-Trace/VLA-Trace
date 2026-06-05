from __future__ import annotations

import numpy as np

from vla_trace.representations.cka import (
    checkpoint_drift_cka,
    layerwise_checkpoint_drift_cka,
    layerwise_cross_modal_cka,
    matched_layer_checkpoint_drift_summary,
    cross_modal_cka_profile,
    linear_cka,
)


def test_linear_cka_identity_is_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 8))
    score = linear_cka(x, x.copy())
    assert np.isclose(score, 1.0)


def test_cross_modal_profile_tracks_related_modalities() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(48, 6))
    profile = cross_modal_cka_profile(
        {
            "vision": base,
            "text": base + 0.01 * rng.normal(size=base.shape),
            "action": rng.normal(size=(48, 6)),
        }
    )
    assert np.isclose(profile["vision"]["vision"], 1.0)
    assert profile["vision"]["text"] > 0.99
    assert profile["vision"]["action"] < profile["vision"]["text"]


def test_checkpoint_drift_report_orders_by_similarity_to_reference() -> None:
    rng = np.random.default_rng(2)
    anchor = rng.normal(size=(40, 5))
    report = checkpoint_drift_cka(
        {
            "step_0": anchor,
            "step_1": anchor + 0.05 * rng.normal(size=anchor.shape),
            "step_2": rng.normal(size=(40, 5)),
        }
    )
    assert report["reference"] == "step_0"
    assert np.isclose(report["drift"]["step_0"], 0.0)
    assert report["drift"]["step_1"] < report["drift"]["step_2"]
    assert report["consecutive"][0]["from"] == "step_0"
    assert report["consecutive"][1]["to"] == "step_2"


def test_layerwise_cross_modal_and_drift() -> None:
    rng = np.random.default_rng(3)
    vision0 = rng.normal(size=(24, 4))
    text0 = vision0 + 0.01 * rng.normal(size=(24, 4))
    bank = {
        "vision_pooled/layer_0": vision0,
        "text_pooled/layer_0": text0,
        "joint_pooled/layer_0": vision0 + text0,
    }
    profile = layerwise_cross_modal_cka(bank)
    assert profile[0] > 0.99

    drift = layerwise_checkpoint_drift_cka(
        bank,
        {"joint_pooled/layer_0": bank["joint_pooled/layer_0"] + 0.01},
    )
    assert drift[0] > 0.99


def test_matched_layer_checkpoint_drift_summary_reports_view_means() -> None:
    rng = np.random.default_rng(4)
    c0 = {}
    c1 = {}
    for view in ("vision_pooled", "text_pooled", "joint_pooled"):
        for layer in (0, 1):
            base = rng.normal(size=(32, 5))
            c0[f"{view}/layer_{layer}"] = base
            c1[f"{view}/layer_{layer}"] = base + 0.01 * rng.normal(size=base.shape)

    summary = matched_layer_checkpoint_drift_summary(
        {"C0": c0, "C1": c1},
        reference="C0",
    )

    assert summary["reference"] == "C0"
    assert summary["targets"]["C1"]["vision_pooled"]["status"] == "ok"
    assert summary["targets"]["C1"]["vision_pooled"]["n_layers"] == 2
    assert summary["targets"]["C1"]["vision_pooled"]["mean_cka"] > 0.99
    assert summary["targets"]["C1"]["joint_pooled"]["layers"] == [0, 1]
