"""Template adapter for OpenVLA/OpenVLA-OFT representation extraction.

Copy this file into your own project and replace the TODO blocks with your
model loading and forward-hook code. VLA-Trace calls `create_adapter(...)`,
then `adapter.extract_hidden_states(row)` for each manifest row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def create_adapter(**kwargs: Any) -> "OpenVLARepresentationAdapter":
    return OpenVLARepresentationAdapter(**kwargs)


class OpenVLARepresentationAdapter:
    """Minimal adapter contract for `vla-trace collect-repr`.

    Required input row fields usually include `image_path` and `instruction`.
    Return a mapping from layer index to a `[tokens, hidden_dim]` array.
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        data_root: str | None = None,
        device: str = "cuda",
        **_: Any,
    ) -> None:
        self.model_path = model_path
        self.data_root = data_root
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load your local OpenVLA/OpenVLA-OFT model here."""

        if not self.model_path:
            raise ValueError("OpenVLA representation adapter requires model_path")
        # TODO: initialize processor/model and register hidden-state hooks.
        # Example shape expected by VLA-Trace after each forward:
        # {0: np.ndarray[tokens, hidden_dim], 1: ..., 31: ...}
        raise NotImplementedError("Replace load() with your model initialization")

    def extract_hidden_states(self, row: dict[str, Any]) -> dict[int, np.ndarray]:
        """Run one manifest sample and return layer hidden states."""

        image_path = Path(str(row["image_path"]))
        instruction = str(row.get("instruction") or row.get("task_description") or "")
        _ = image_path, instruction
        # TODO: load image, run your model forward with hidden states enabled,
        # then return {layer_index: hidden[layer_index]}.
        raise NotImplementedError("Return {layer: [tokens, hidden_dim]} arrays")
