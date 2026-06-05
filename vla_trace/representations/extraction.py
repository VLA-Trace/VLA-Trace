"""Stage 1 representation-bank collection from manifests and adapter traces."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from vla_trace.io.registry import DEFAULT_TOKEN_LAYOUTS, normalize_model
from vla_trace.representations.bank import RepresentationBank, load_representation_bank, save_representation_bank
from vla_trace.representations.manifest import read_manifest


AdapterFactory = Callable[..., Any]


def collect_representation_bank_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_path = cfg.get("manifest_path")
    output_path = cfg.get("output_path") or cfg.get("bank_output")
    if not manifest_path:
        raise ValueError("collect-repr requires manifest_path or --manifest")
    if not output_path:
        raise ValueError("collect-repr requires output_path or --bank-output")

    rows = read_manifest(manifest_path, max_samples=_optional_int(cfg.get("max_samples")))
    adapter = _load_adapter(cfg) if cfg.get("adapter") else None
    groups = resolve_token_groups(cfg)
    bank = collect_representation_bank(
        rows,
        token_groups=groups,
        hidden_state_dir=cfg.get("hidden_state_dir"),
        adapter=adapter,
        metadata=_metadata_from_config(cfg, manifest_path),
    )
    output = save_representation_bank(output_path, bank)
    return {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "output_path": str(output),
        "n_samples": len(rows),
        "array_keys": sorted(bank.arrays),
        "metadata": bank.metadata,
    }


def convert_bank(input_path: str | Path, output_path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    bank = load_representation_bank(input_path)
    merged = dict(bank.metadata)
    if metadata:
        merged.update(dict(metadata))
    return save_representation_bank(output_path, RepresentationBank(arrays=bank.arrays, metadata=merged))


def collect_representation_bank(
    rows: list[dict[str, Any]],
    *,
    token_groups: Mapping[str, Any],
    hidden_state_dir: str | Path | None = None,
    adapter: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RepresentationBank:
    if not rows:
        raise ValueError("representation collection requires at least one manifest row")
    collected: dict[str, list[np.ndarray]] = {}
    sample_ids: list[str] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", index))
        hidden = _extract_hidden_states(row, hidden_state_dir=hidden_state_dir, adapter=adapter)
        pooled = pool_hidden_state_views(hidden, token_groups=token_groups)
        for key, value in pooled.items():
            collected.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
        sample_ids.append(sample_id)
    arrays = {key: np.stack(values, axis=0) for key, values in collected.items()}
    meta = dict(metadata or {})
    meta.setdefault("sample_ids", sample_ids)
    meta.setdefault("token_groups", _serializable_token_groups(token_groups))
    return RepresentationBank(arrays=arrays, metadata=meta)


def pool_hidden_state_views(hidden_states: Mapping[str | int, Any], *, token_groups: Mapping[str, Any]) -> dict[str, np.ndarray]:
    pooled: dict[str, np.ndarray] = {}
    layers = _normalize_hidden_state_layers(hidden_states)
    selectors = {name: _parse_selector(selector) for name, selector in token_groups.items()}
    for layer, tokens in sorted(layers.items()):
        token_array = _normalize_token_array(tokens)
        for view_name, selector in selectors.items():
            group = token_array[selector]
            if group.shape[0] == 0:
                raise ValueError(f"Token group {view_name!r} is empty for layer {layer}")
            pooled[f"{view_name}/layer_{layer}"] = group.mean(axis=0)
    return pooled


def resolve_token_groups(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if cfg.get("token_groups"):
        return dict(cfg["token_groups"])
    layout = dict(cfg.get("token_layout") or {})
    family = _family_from_config(cfg)
    if not layout and family in DEFAULT_TOKEN_LAYOUTS:
        layout = dict(DEFAULT_TOKEN_LAYOUTS[family])
    visual_tokens = int(layout.get("visual_tokens", 0))
    text_tokens = int(layout.get("text_tokens", 0))
    if visual_tokens <= 0 or text_tokens <= 0:
        raise ValueError("collect-repr requires token_groups or visual/text token counts")
    if family == "openvla":
        prefix_tokens = int(layout.get("prefix_tokens", 1))
        image_start = prefix_tokens
    else:
        image_start = 0
    image_end = image_start + visual_tokens
    text_start = image_end
    text_end = text_start + text_tokens
    return {
        "vision_pooled": [image_start, image_end],
        "text_pooled": [text_start, text_end],
        "joint_pooled": [image_start, text_end],
    }


def _extract_hidden_states(
    row: dict[str, Any],
    *,
    hidden_state_dir: str | Path | None,
    adapter: Any | None,
) -> Mapping[str | int, Any]:
    if adapter is not None:
        hidden = adapter.extract_hidden_states(row)
        if not isinstance(hidden, Mapping):
            raise ValueError("adapter.extract_hidden_states(row) must return a mapping")
        return hidden
    path = row.get("hidden_states_path")
    if not path and hidden_state_dir:
        sample_id = str(row.get("sample_id"))
        for suffix in (".npz", ".npy", ".json"):
            candidate = Path(hidden_state_dir) / f"{sample_id}{suffix}"
            if candidate.exists():
                path = str(candidate)
                break
    if not path:
        raise ValueError(
            "Manifest row is missing hidden_states_path and no adapter was provided. "
            "Run your model adapter to export hidden states, or pass --adapter module:function."
        )
    return load_hidden_state_artifact(path)


def load_hidden_state_artifact(path: str | Path) -> Mapping[str | int, np.ndarray]:
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if "hidden_states" in payload:
                return _split_layer_stack(np.asarray(payload["hidden_states"]))
            return {name: np.asarray(payload[name]) for name in payload.files}
    if source.suffix == ".npy":
        return _split_layer_stack(np.asarray(np.load(source, allow_pickle=False)))
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if "hidden_states" in payload:
            return _split_layer_stack(np.asarray(payload["hidden_states"]))
        return {name: np.asarray(value) for name, value in dict(payload).items()}
    raise ValueError(f"Unsupported hidden-state artifact format: {source.suffix}")


def _load_adapter(cfg: Mapping[str, Any]) -> Any:
    spec = str(cfg["adapter"])
    if ":" not in spec:
        raise ValueError("adapter must be an import path like package.module:create_adapter")
    module_name, function_name = spec.split(":", 1)
    factory: AdapterFactory = getattr(importlib.import_module(module_name), function_name)
    kwargs = dict(cfg.get("adapter_kwargs") or {})
    for key in ("model_path", "data_root", "model", "benchmark"):
        if key in cfg:
            kwargs.setdefault(key, cfg[key])
    adapter = factory(**kwargs)
    if hasattr(adapter, "load"):
        adapter.load()
    return adapter


def _normalize_hidden_state_layers(hidden_states: Mapping[str | int, Any]) -> dict[int, np.ndarray]:
    normalized: dict[int, np.ndarray] = {}
    for name, value in hidden_states.items():
        text = str(name)
        if text.startswith("layer_"):
            text = text.removeprefix("layer_")
        elif text.startswith("hidden_state_"):
            text = text.removeprefix("hidden_state_")
        try:
            layer = int(text)
        except ValueError:
            continue
        normalized[layer] = np.asarray(value)
    if not normalized:
        raise ValueError("No layer hidden states found. Use keys like layer_0 or a hidden_states stack.")
    return normalized


def _split_layer_stack(array: np.ndarray) -> dict[int, np.ndarray]:
    values = np.asarray(array)
    if values.ndim < 3:
        raise ValueError("hidden_states stack must have shape [layers, tokens, dim] or [layers, batch, tokens, dim]")
    return {layer: values[layer] for layer in range(values.shape[0])}


def _normalize_token_array(tokens: Any) -> np.ndarray:
    array = np.asarray(tokens, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            array = array.mean(axis=0)
        else:
            array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected token hidden states with shape [tokens, dim], got {array.shape}")
    return array


def _parse_selector(selector: Any) -> slice | list[int]:
    if isinstance(selector, slice):
        return selector
    if isinstance(selector, str):
        if selector == "all":
            return slice(None)
        if ":" in selector:
            start_text, stop_text = selector.split(":", 1)
            start = int(start_text) if start_text else None
            stop = int(stop_text) if stop_text else None
            return slice(start, stop)
        return [int(piece.strip()) for piece in selector.split(",") if piece.strip()]
    if isinstance(selector, (list, tuple)) and len(selector) == 2 and all(
        isinstance(item, (int, type(None))) for item in selector
    ):
        return slice(selector[0], selector[1])
    if isinstance(selector, (list, tuple)):
        return [int(item) for item in selector]
    raise ValueError(f"Unsupported token selector: {selector!r}")


def _serializable_token_groups(token_groups: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, selector in token_groups.items():
        if isinstance(selector, slice):
            out[name] = [selector.start, selector.stop]
        else:
            out[name] = selector
    return out


def _metadata_from_config(cfg: Mapping[str, Any], manifest_path: Any) -> dict[str, Any]:
    meta = {
        key: cfg[key]
        for key in (
            "model",
            "benchmark",
            "model_path",
            "data_root",
            "checkpoint_name",
            "prompt_style",
        )
        if key in cfg
    }
    meta["manifest_path"] = str(manifest_path)
    return meta


def _family_from_config(cfg: Mapping[str, Any]) -> str:
    family = cfg.get("family")
    if family:
        return normalize_model(str(family))
    model = str(cfg.get("model", "")).lower()
    if "pi05" in model or "pi0" in model:
        return "pi05"
    return "openvla"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
