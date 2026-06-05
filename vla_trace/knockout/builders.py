"""Public additive attention-mask builders for Stage 2 knockout runs."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from vla_trace.knockout.specs import KnockoutSpec, TokenPartition, TokenPartitions


@dataclass(frozen=True)
class AdditiveMask:
    """Layer-major additive attention mask."""

    values: np.ndarray
    layers: tuple[int, ...]

    def at(self, layer: int, query: int, key: int) -> float:
        return float(self.values[self.layers.index(layer), query, key])


def make_openvla_partitions(
    *,
    visual_tokens: int,
    text_tokens: int,
    action_tokens: int,
    prefix_tokens: int = 1,
) -> TokenPartitions:
    prefix = TokenPartition("prefix", 0, prefix_tokens)
    visual = TokenPartition("visual", prefix.stop, prefix.stop + visual_tokens)
    text = TokenPartition("text", visual.stop, visual.stop + text_tokens)
    action = TokenPartition("action", text.stop, text.stop + action_tokens)
    return TokenPartitions(prefix=prefix, visual=visual, text=text, action=action)


def make_pi05_partitions(
    *,
    visual_tokens: int,
    text_tokens: int,
    action_tokens: int,
    token_order: str = "text,visual,action",
) -> TokenPartitions:
    """Build pi0.5 partitions.

    `token_order` lets adapters match the model's actual packed sequence. The
    public default follows the paper prefill matrix convention. If an adapter
    exports a visual-first packed sequence, pass `visual,text,action`.
    """
    sizes = {
        "visual": int(visual_tokens),
        "text": int(text_tokens),
        "action": int(action_tokens),
    }
    parts = _ordered_partitions(token_order, sizes)
    return TokenPartitions(prefix=None, visual=parts["visual"], text=parts["text"], action=parts["action"])


def _ordered_partitions(token_order: str, sizes: dict[str, int]) -> dict[str, TokenPartition]:
    order = tuple(piece.strip().lower() for piece in token_order.split(",") if piece.strip())
    expected = tuple(sizes)
    if set(order) != set(expected) or len(order) != len(expected):
        raise ValueError(f"token_order must contain exactly: {', '.join(expected)}")
    cursor = 0
    parts: dict[str, TokenPartition] = {}
    for name in order:
        stop = cursor + sizes[name]
        parts[name] = TokenPartition(name, cursor, stop)
        cursor = stop
    return parts


def build_additive_mask(spec: KnockoutSpec, partitions: TokenPartitions) -> AdditiveMask:
    """Build a public additive mask for the requested family and phase."""

    layers = spec.layers or (0,)
    mask = np.zeros((len(layers), partitions.total_tokens, partitions.total_tokens), dtype=np.float32)
    if "baseline" in spec.modes:
        return AdditiveMask(values=mask, layers=layers)
    if spec.direction:
        _apply_directional(mask, spec, partitions)
        return AdditiveMask(values=mask, layers=layers)
    if _is_combined_mode(spec.mode):
        for phase, mode in _combined_phase_modes(spec.family, spec.mode).items():
            phase_spec = replace(spec, phase=phase, mode=mode)
            if spec.family == "openvla":
                _apply_openvla(mask, phase_spec, partitions)
            elif spec.family == "pi05":
                _apply_pi05(mask, phase_spec, partitions)
        return AdditiveMask(values=mask, layers=layers)

    if spec.family == "openvla":
        _apply_openvla(mask, spec, partitions)
    elif spec.family == "pi05":
        _apply_pi05(mask, spec, partitions)
    else:
        raise ValueError(f"Unsupported family: {spec.family}")
    return AdditiveMask(values=mask, layers=layers)


def _block(mask: np.ndarray, value: float, q: TokenPartition, k: TokenPartition) -> None:
    mask[:, q.start:q.stop, k.start:k.stop] = value


def _block_many(mask: np.ndarray, value: float, q: TokenPartition, keys: tuple[TokenPartition, ...]) -> None:
    for key in keys:
        if key.size > 0:
            _block(mask, value, q, key)


def _text_key_partitions(partitions: TokenPartitions, text_scope: str) -> tuple[TokenPartition, ...]:
    """Return text/structural key spans selected by a public text scope.

    The public configs do not know the exact tokenizer offsets for every
    prompt template, so we use a conservative convention: the final text token
    is the newline/suffix token when present, and the optional prefix partition
    represents BOS or prompt prefix structure.
    """
    text = partitions.text
    instruction_stop = max(text.start, text.stop - 1)
    instruction = TokenPartition("instruction", text.start, instruction_stop)
    newline = TokenPartition("newline", instruction_stop, text.stop)
    prefix_parts = (partitions.prefix,) if partitions.prefix is not None else ()
    if text_scope in {"instruction", "semantic_instruction", "exclude_newline"}:
        return (instruction,)
    if text_scope == "newline_only":
        return (newline,)
    if text_scope == "bos_newline":
        return (*prefix_parts, newline)
    if text_scope in {"all", "full"}:
        return (*prefix_parts, text)
    raise ValueError(f"Unsupported text scope: {text_scope}")


def _apply_openvla(mask: np.ndarray, spec: KnockoutSpec, partitions: TokenPartitions) -> None:
    text_keys = _text_key_partitions(partitions, spec.text_scope)
    for mode in spec.modes:
        if mode == "no_fusion":
            if spec.includes_phase("prefill"):
                for text_key in text_keys:
                    _block(mask, spec.additive_block_value, text_key, partitions.visual)
            continue
        if mode in {"no_image", "no_vl"}:
            if spec.includes_phase("prefill"):
                for text_key in text_keys:
                    _block(mask, spec.additive_block_value, text_key, partitions.visual)
                _block(mask, spec.additive_block_value, partitions.action, partitions.visual)
            if spec.includes_phase("generation"):
                _block(mask, spec.additive_block_value, partitions.action, partitions.visual)
        if mode == "no_text":
            if spec.includes_phase("prefill"):
                _block_many(mask, spec.additive_block_value, partitions.action, text_keys)
            if spec.includes_phase("generation"):
                _block_many(mask, spec.additive_block_value, partitions.action, text_keys)
        if mode == "no_vl" and spec.includes_phase("generation"):
            _block_many(mask, spec.additive_block_value, partitions.action, text_keys)


def _apply_pi05(mask: np.ndarray, spec: KnockoutSpec, partitions: TokenPartitions) -> None:
    text_keys = _text_key_partitions(partitions, spec.text_scope)
    for mode in spec.modes:
        if mode in {"no_text", "no_image", "no_vl", "no_fusion"} and spec.includes_phase("prefill"):
            _block_many(mask, spec.additive_block_value, partitions.visual, text_keys)
            for text_key in text_keys:
                _block(mask, spec.additive_block_value, text_key, partitions.visual)
        if mode == "no_fusion":
            continue
        if mode == "no_text" and spec.includes_phase("generation"):
            _block_many(mask, spec.additive_block_value, partitions.action, text_keys)
        if mode == "no_image" and spec.includes_phase("generation"):
            _block(mask, spec.additive_block_value, partitions.action, partitions.visual)
        if mode == "no_vl" and spec.includes_phase("generation"):
            _block_many(mask, spec.additive_block_value, partitions.action, text_keys)
            _block(mask, spec.additive_block_value, partitions.action, partitions.visual)


def _apply_directional(mask: np.ndarray, spec: KnockoutSpec, partitions: TokenPartitions) -> None:
    direction = str(spec.direction)
    value = spec.additive_block_value
    if direction in {"image<->text", "text<->image"}:
        if spec.includes_phase("prefill") or spec.includes_phase("generation"):
            for text_key in _text_key_partitions(partitions, spec.text_scope):
                _block(mask, value, text_key, partitions.visual)
                _block(mask, value, partitions.visual, text_key)
        return
    source, target = direction.split("->", 1)
    key_parts = _direction_source_partitions(source, partitions, spec.text_scope)
    query_part = _direction_target_partition(target, partitions)
    if target == "action" and not spec.includes_phase("generation"):
        return
    if target in {"image", "text"} and not (spec.includes_phase("prefill") or spec.includes_phase("generation")):
        return
    _block_many(mask, value, query_part, key_parts)


def _direction_source_partitions(source: str, partitions: TokenPartitions, text_scope: str) -> tuple[TokenPartition, ...]:
    if source == "image":
        return (partitions.visual,)
    if source == "text":
        return _text_key_partitions(partitions, text_scope)
    if source == "action":
        return (partitions.action,)
    raise ValueError(f"Unsupported direction source: {source}")


def _direction_target_partition(target: str, partitions: TokenPartitions) -> TokenPartition:
    if target == "image":
        return partitions.visual
    if target == "text":
        return partitions.text
    if target == "action":
        return partitions.action
    raise ValueError(f"Unsupported direction target: {target}")


def _is_combined_mode(mode: str) -> bool:
    return "__generation_" in mode and "_prefill_" in mode


def _combined_phase_modes(family: str, mode: str) -> dict[str, str]:
    if family == "pi05":
        mapping = {
            "pi05_prefill_no_vl__generation_no_text": {"prefill": "no_vl", "generation": "no_text"},
            "pi05_prefill_no_vl__generation_no_image": {"prefill": "no_vl", "generation": "no_image"},
        }
    elif family == "openvla":
        mapping = {
            "openvla_prefill_no_image__generation_no_text": {"prefill": "no_image", "generation": "no_text"},
            "openvla_prefill_no_image__generation_no_image": {"prefill": "no_image", "generation": "no_image"},
        }
    else:
        mapping = {}
    if mode not in mapping:
        raise ValueError(f"Unsupported combined knockout mode for {family}: {mode}")
    return mapping[mode]
