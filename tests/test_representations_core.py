from __future__ import annotations

import numpy as np

from vla_trace.representations.bank import load_representation_bank, save_representation_bank
from vla_trace.representations.extraction import collect_representation_bank
from vla_trace.representations.pooling import TokenLayout, pool_tokens


def test_pool_tokens_respects_layout_groups() -> None:
    tokens = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)
    layout = TokenLayout(groups={"vision": slice(0, 2), "action": [2, 3, 4]})
    pooled = pool_tokens(tokens, layout)
    assert set(pooled) == {"vision", "action"}
    np.testing.assert_allclose(pooled["vision"], tokens[:, 0:2, :].mean(axis=1))
    np.testing.assert_allclose(pooled["action"], tokens[:, [2, 3, 4], :].mean(axis=1))


def test_representation_bank_round_trip_json_and_npz(tmp_path) -> None:
    arrays = {
        "vision": np.arange(12, dtype=np.float64).reshape(3, 4),
        "text": np.eye(3, dtype=np.float64),
    }
    metadata = {"stage": "stage1", "kind": "unit"}
    for suffix in (".json", ".npz"):
        target = tmp_path / f"bank{suffix}"
        save_representation_bank(target, arrays, metadata=metadata)
        loaded = load_representation_bank(target)
        assert loaded.metadata == metadata
        for name, expected in arrays.items():
            np.testing.assert_allclose(loaded.arrays[name], expected)


def test_load_legacy_nested_representation_bank(tmp_path) -> None:
    target = tmp_path / "legacy.json"
    target.write_text(
        """
{
  "sample_ids": ["a", "b"],
  "representations": {
    "vision_pooled": {"0": [[1, 0], [0, 1]]},
    "text_pooled": {"0": [[1, 0], [0, 1]]}
  }
}
""",
        encoding="utf-8",
    )
    loaded = load_representation_bank(target)
    assert loaded.metadata["sample_ids"] == ["a", "b"]
    assert "vision_pooled/layer_0" in loaded.arrays
    np.testing.assert_allclose(loaded.arrays["text_pooled/layer_0"], [[1, 0], [0, 1]])


def test_collect_representation_bank_from_hidden_state_rows() -> None:
    rows = [{"sample_id": "s0", "hidden_states_path": ""}, {"sample_id": "s1", "hidden_states_path": ""}]
    hidden0 = {
        0: np.arange(5 * 3, dtype=np.float32).reshape(5, 3),
        1: np.ones((5, 3), dtype=np.float32),
    }
    hidden1 = {
        0: np.arange(5 * 3, dtype=np.float32).reshape(5, 3) + 10,
        1: np.full((5, 3), 2.0, dtype=np.float32),
    }

    class Adapter:
        def __init__(self) -> None:
            self.index = 0

        def extract_hidden_states(self, _row):
            value = [hidden0, hidden1][self.index]
            self.index += 1
            return value

    bank = collect_representation_bank(
        rows,
        token_groups={"vision_pooled": [0, 2], "text_pooled": [2, 5], "joint_pooled": [0, 5]},
        adapter=Adapter(),
        metadata={"checkpoint": "C1"},
    )

    assert bank.metadata["sample_ids"] == ["s0", "s1"]
    assert bank.arrays["vision_pooled/layer_0"].shape == (2, 3)
    np.testing.assert_allclose(bank.arrays["text_pooled/layer_1"], [[1, 1, 1], [2, 2, 2]])
