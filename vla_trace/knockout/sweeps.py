"""Config-to-spec helpers for Stage 2 knockout sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from vla_trace.knockout.builders import make_openvla_partitions, make_pi05_partitions
from vla_trace.knockout.specs import KnockoutSpec, LayerWindow, expand_layer_window


@dataclass(frozen=True)
class KnockoutSweep:
    """Serializable Stage 2 sweep summary."""

    spec: KnockoutSpec
    num_layers: int
    token_layout: dict[str, Any]
    phase_specs: dict[str, KnockoutSpec] = field(default_factory=dict)
    setting: str = ""

    def summary(self) -> str:
        suffix = f" setting={self.setting}" if self.setting else ""
        return (
            f"knockout family={self.spec.family} phase={self.spec.phase} "
            f"mode={self.spec.mode} layers={len(self.spec.layers)}{suffix}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"]["layers"] = list(self.spec.layers)
        payload["phase_specs"] = {
            phase: {**asdict(spec), "layers": list(spec.layers)}
            for phase, spec in self.phase_specs.items()
        }
        return payload


def build_sweep_from_config(cfg: dict[str, Any]) -> KnockoutSweep:
    family = str(cfg["family"])
    phase = str(cfg["phase"])
    mode = str(cfg["mode"])
    text_scope = str(cfg.get("text_scope", "semantic_instruction"))
    num_layers = int(cfg["num_layers"])
    layers = _resolve_layers(cfg.get("layers", {}), num_layers)
    direction = cfg.get("direction")
    spec = KnockoutSpec(
        family=family,
        phase=phase,
        mode=mode,
        layers=layers,
        text_scope=text_scope,
        direction=str(direction) if direction else None,
    )
    phase_specs = _build_phase_specs(cfg, spec)

    token_layout = cfg.get("token_layout", {})
    visual_tokens = int(token_layout["visual_tokens"])
    text_tokens = int(token_layout["text_tokens"])
    action_tokens = int(token_layout["action_tokens"])
    token_order = str(token_layout.get("token_order", _default_token_order(family)))
    if family == "openvla":
        make_openvla_partitions(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            action_tokens=action_tokens,
        )
    elif family == "pi05":
        make_pi05_partitions(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            action_tokens=action_tokens,
            token_order=token_order,
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    return KnockoutSweep(
        spec=spec,
        num_layers=num_layers,
        token_layout={
            "visual_tokens": visual_tokens,
            "text_tokens": text_tokens,
            "action_tokens": action_tokens,
            **({"token_order": token_order} if family == "pi05" else {}),
        },
        phase_specs=phase_specs,
        setting=str(cfg.get("setting", "")),
    )


def _resolve_layers(layers_cfg: dict[str, Any], num_layers: int) -> tuple[int, ...]:
    layer_type = str(layers_cfg.get("type", "explicit"))
    if layer_type == "all":
        return tuple(range(num_layers))
    if layer_type == "window":
        window = LayerWindow(
            center_layers=tuple(int(layer) for layer in layers_cfg.get("center_layers", [])),
            window_size=int(layers_cfg["window_size"]),
        )
        return window.expand(num_layers)
    if layer_type == "explicit":
        if "values" in layers_cfg:
            values = tuple(int(layer) for layer in layers_cfg["values"])
        else:
            values = tuple(range(num_layers))
        return expand_layer_window(num_layers, values, 1)
    if layer_type == "scan":
        start = int(layers_cfg.get("start", 0))
        stop = int(layers_cfg.get("stop", num_layers))
        if start < 0 or stop > num_layers or stop <= start:
            raise ValueError("scan layers must satisfy 0 <= start < stop <= num_layers")
        return tuple(range(start, stop))
    raise ValueError(f"Unsupported layer selection type: {layer_type}")


def _build_phase_specs(cfg: dict[str, Any], spec: KnockoutSpec) -> dict[str, KnockoutSpec]:
    policy = cfg.get("phase_policy") or {}
    phase_specs: dict[str, KnockoutSpec] = {}
    if isinstance(policy, dict):
        for phase in ("prefill", "generation"):
            phase_cfg = policy.get(phase)
            if isinstance(phase_cfg, dict) and phase_cfg.get("mode"):
                phase_specs[phase] = KnockoutSpec(
                    family=spec.family,
                    phase=phase,
                    mode=str(phase_cfg["mode"]),
                    layers=spec.layers,
                    text_scope=str(phase_cfg.get("text_scope", spec.text_scope)),
                    direction=phase_cfg.get("direction"),
                )
    prefill_mode = cfg.get("prefill_mode")
    generation_mode = cfg.get("generation_mode")
    if prefill_mode:
        phase_specs["prefill"] = KnockoutSpec(
            family=spec.family,
            phase="prefill",
            mode=str(prefill_mode),
            layers=spec.layers,
            text_scope=spec.text_scope,
        )
    if generation_mode:
        phase_specs["generation"] = KnockoutSpec(
            family=spec.family,
            phase="generation",
            mode=str(generation_mode),
            layers=spec.layers,
            text_scope=spec.text_scope,
        )
    if spec.mode == "pi05_prefill_no_vl__generation_no_text":
        phase_specs.update(
            {
                "prefill": KnockoutSpec("pi05", "prefill", "no_vl", spec.layers, spec.text_scope),
                "generation": KnockoutSpec("pi05", "generation", "no_text", spec.layers, spec.text_scope),
            }
        )
    elif spec.mode == "pi05_prefill_no_vl__generation_no_image":
        phase_specs.update(
            {
                "prefill": KnockoutSpec("pi05", "prefill", "no_vl", spec.layers, spec.text_scope),
                "generation": KnockoutSpec("pi05", "generation", "no_image", spec.layers, spec.text_scope),
            }
        )
    elif spec.mode == "openvla_prefill_no_image__generation_no_text":
        phase_specs.update(
            {
                "prefill": KnockoutSpec("openvla", "prefill", "no_image", spec.layers, spec.text_scope),
                "generation": KnockoutSpec("openvla", "generation", "no_text", spec.layers, spec.text_scope),
            }
        )
    elif spec.mode == "openvla_prefill_no_image__generation_no_image":
        phase_specs.update(
            {
                "prefill": KnockoutSpec("openvla", "prefill", "no_image", spec.layers, spec.text_scope),
                "generation": KnockoutSpec("openvla", "generation", "no_image", spec.layers, spec.text_scope),
            }
        )
    return phase_specs


def _default_token_order(family: str) -> str:
    if family == "pi05":
        return "text,visual,action"
    return "visual,text,action"
