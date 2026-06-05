"""Public model-adapter contract.

The open-source core keeps model execution behind adapters. Stage 1/2 analysis
code can run on saved traces without importing heavy model packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vla_trace.io import load_config


@dataclass(frozen=True)
class TokenLayoutSpec:
    """Static token-layout metadata for model-family docs and smoke checks."""

    visual_tokens: int
    action_tokens: int | None = None
    structural_tokens: list[str] | None = None
    text_scope: str = "semantic_instruction"


@dataclass(frozen=True)
class ModelSpec:
    """Declarative model adapter metadata loaded from `configs/models/*.yaml`."""

    name: str
    family: str
    adapter: str
    checkpoint: str | None
    token_layout: TokenLayoutSpec
    supported_stages: list[str]
    notes: str = ""

    @property
    def supports_stage1(self) -> bool:
        return "stage1" in self.supported_stages

    @property
    def supports_stage2(self) -> bool:
        return "stage2" in self.supported_stages

    @property
    def supports_stage3(self) -> bool:
        return "stage3" in self.supported_stages


def load_model_spec(path: str | Path) -> ModelSpec:
    cfg = load_config(path)
    layout_cfg = cfg.get("token_layout", {})
    layout = TokenLayoutSpec(
        visual_tokens=int(layout_cfg.get("visual_tokens", 256)),
        action_tokens=layout_cfg.get("action_tokens"),
        structural_tokens=list(layout_cfg.get("structural_tokens", [])),
        text_scope=str(layout_cfg.get("text_scope", "semantic_instruction")),
    )
    if layout.action_tokens is not None:
        layout = TokenLayoutSpec(
            visual_tokens=layout.visual_tokens,
            action_tokens=int(layout.action_tokens),
            structural_tokens=layout.structural_tokens,
            text_scope=layout.text_scope,
        )
    return ModelSpec(
        name=str(cfg["name"]),
        family=str(cfg["family"]),
        adapter=str(cfg["adapter"]),
        checkpoint=cfg.get("checkpoint"),
        token_layout=layout,
        supported_stages=list(cfg.get("supported_stages", [])),
        notes=str(cfg.get("notes", "")),
    )


class ModelAdapter:
    """Execution adapter interface for future heavy model integrations."""

    spec: ModelSpec

    def load(self) -> None:
        raise NotImplementedError

    def extract_hidden_states(self, sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run_knockout(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
