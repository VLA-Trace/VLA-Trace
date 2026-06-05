"""Stage 2 knockout specifications and token partitions."""

from __future__ import annotations

from dataclasses import dataclass, field


SUPPORTED_FAMILIES = frozenset({"openvla", "pi05"})
SUPPORTED_PHASES = frozenset({"prefill", "generation", "both"})
SUPPORTED_TEXT_SCOPES = frozenset(
    {
        "all",
        "instruction",
        "semantic_instruction",
        "full",
        "bos_newline",
        "newline_only",
        "exclude_newline",
    }
)
SUPPORTED_MODES = frozenset(
    {
        "baseline",
        "no_image",
        "no_text",
        "no_vl",
        "no_fusion",
        "pi05_prefill_no_vl__generation_no_text",
        "pi05_prefill_no_vl__generation_no_image",
        "openvla_prefill_no_image__generation_no_text",
        "openvla_prefill_no_image__generation_no_image",
    }
)
SUPPORTED_DIRECTIONS = frozenset(
    {
        "image->action",
        "text->action",
        "image->text",
        "text->image",
        "image<->text",
        "text<->image",
    }
)


def _normalize_modes(mode: str) -> tuple[str, ...]:
    pieces = tuple(part.strip() for part in mode.split("+") if part.strip())
    if not pieces:
        raise ValueError("Expected at least one knockout mode")
    unknown = set(pieces) - SUPPORTED_MODES
    if unknown:
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported knockout mode(s): {unknown_str}")
    if "baseline" in pieces and len(pieces) > 1:
        raise ValueError("baseline cannot be combined with other knockout modes")
    return tuple(dict.fromkeys(pieces))


@dataclass(frozen=True)
class TokenPartition:
    """Half-open token span `[start, stop)` for a semantic partition."""

    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"{self.name} start must be >= 0")
        if self.stop < self.start:
            raise ValueError(f"{self.name} stop must be >= start")

    @property
    def size(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class TokenPartitions:
    """Family-specific token spans used by public Stage 2 mask builders."""

    visual: TokenPartition
    text: TokenPartition
    action: TokenPartition
    prefix: TokenPartition | None = None
    extra: tuple[TokenPartition, ...] = ()

    def __post_init__(self) -> None:
        ordered = sorted(
            [part for part in (self.prefix, self.visual, self.text, self.action, *self.extra) if part is not None],
            key=lambda part: part.start,
        )
        cursor = 0
        for part in ordered:
            if part.start != cursor:
                raise ValueError(f"Partition {part.name} must start at {cursor}, got {part.start}")
            cursor = part.stop

    @property
    def total_tokens(self) -> int:
        return max(
            part.stop
            for part in (self.prefix, self.visual, self.text, self.action, *self.extra)
            if part is not None
        )


@dataclass(frozen=True)
class LayerWindow:
    """Window expansion spec for layer targeting."""

    center_layers: tuple[int, ...]
    window_size: int

    def expand(self, num_layers: int) -> tuple[int, ...]:
        return expand_layer_window(
            num_layers=num_layers,
            center_layers=self.center_layers,
            window_size=self.window_size,
        )


@dataclass(frozen=True)
class KnockoutSpec:
    """Validated Stage 2 knockout request."""

    family: str
    phase: str
    mode: str
    layers: tuple[int, ...] = field(default_factory=tuple)
    text_scope: str = "semantic_instruction"
    additive_block_value: float = float("-inf")
    direction: str | None = None

    def __post_init__(self) -> None:
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError(f"Unsupported family: {self.family}")
        if self.phase not in SUPPORTED_PHASES:
            raise ValueError(f"Unsupported phase: {self.phase}")
        if self.text_scope not in SUPPORTED_TEXT_SCOPES:
            raise ValueError(f"Unsupported text scope: {self.text_scope}")
        _normalize_modes(self.mode)
        if self.direction is not None and self.direction not in SUPPORTED_DIRECTIONS:
            raise ValueError(f"Unsupported direction: {self.direction}")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("Layer indices must be >= 0")

    @property
    def modes(self) -> tuple[str, ...]:
        return _normalize_modes(self.mode)

    def includes_phase(self, phase: str) -> bool:
        return self.phase == "both" or self.phase == phase


def expand_layer_window(num_layers: int, center_layers: tuple[int, ...], window_size: int) -> tuple[int, ...]:
    """Expand center layers into a unique sorted window-clipped layer list."""

    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if not center_layers:
        return ()
    if any(layer < 0 or layer >= num_layers for layer in center_layers):
        raise ValueError("center_layers must fall inside [0, num_layers)")

    lower_radius = (window_size - 1) // 2
    upper_radius = window_size // 2
    layers: set[int] = set()
    for center in center_layers:
        start = max(0, center - lower_radius)
        stop = min(num_layers, center + upper_radius + 1)
        layers.update(range(start, stop))
    return tuple(sorted(layers))
