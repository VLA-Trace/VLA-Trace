from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


@dataclass(frozen=True)
class RepresentationBank:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_serializable(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "arrays": {name: array.tolist() for name, array in self.arrays.items()},
        }


def save_representation_bank(
    path: str | Path,
    bank: RepresentationBank | Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    if isinstance(bank, RepresentationBank):
        arrays = {name: _to_numpy(array) for name, array in bank.arrays.items()}
        metadata_map = dict(bank.metadata)
    else:
        arrays = {name: _to_numpy(array) for name, array in bank.items()}
        metadata_map = {}
    if metadata is not None:
        metadata_map.update(dict(metadata))

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".json":
        target.write_text(
            json.dumps(
                {
                    "metadata": metadata_map,
                    "arrays": {name: array.tolist() for name, array in arrays.items()},
                },
                indent=2,
                allow_nan=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return target
    if target.suffix == ".npz":
        np.savez_compressed(
            target,
            __metadata__=json.dumps(metadata_map),
            **arrays,
        )
        return target
    raise ValueError(f"unsupported bank format: {target.suffix}")


def load_representation_bank(path: str | Path) -> RepresentationBank:
    source = Path(path)
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata", {}))
        if "arrays" in payload:
            arrays = {
                name: np.asarray(value)
                for name, value in payload.get("arrays", {}).items()
            }
            return RepresentationBank(arrays=arrays, metadata=metadata)
        if "representations" in payload:
            arrays = _flatten_representation_payload(payload["representations"])
            metadata.update(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {"representations", "arrays"}
                }
            )
            return RepresentationBank(arrays=arrays, metadata=metadata)
        raise ValueError("JSON bank must contain either 'arrays' or 'representations'")
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["__metadata__"])) if "__metadata__" in payload else {}
            arrays = {
                name: np.asarray(payload[name])
                for name in payload.files
                if name != "__metadata__"
            }
        return RepresentationBank(arrays=arrays, metadata=metadata)
    if source.suffix == ".pt":
        return _load_legacy_torch_bank(source)
    raise ValueError(f"unsupported bank format: {source.suffix}")


def flatten_representation_payload(representations: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return _flatten_representation_payload(representations)


def _flatten_representation_payload(representations: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for view_name, value in representations.items():
        if isinstance(value, Mapping):
            for layer, matrix in value.items():
                layer_text = str(layer)
                if layer_text.startswith("layer_"):
                    layer_text = layer_text.removeprefix("layer_")
                arrays[f"{view_name}/layer_{int(layer_text)}"] = _to_numpy(matrix)
        else:
            arrays[str(view_name)] = _to_numpy(value)
    return arrays


def _load_legacy_torch_bank(path: Path) -> RepresentationBank:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional torch install
        raise RuntimeError(
            "Loading legacy .pt representation banks requires torch. "
            "Install torch or convert the bank in an environment that has torch."
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Legacy .pt bank must contain a mapping payload")
    metadata = {
        str(key): _metadata_value(value)
        for key, value in payload.items()
        if key not in {"arrays", "representations"}
    }
    if "arrays" in payload:
        arrays = {str(name): _to_numpy(value) for name, value in dict(payload["arrays"]).items()}
        return RepresentationBank(arrays=arrays, metadata=metadata)
    if "representations" in payload:
        return RepresentationBank(
            arrays=_flatten_representation_payload(payload["representations"]),
            metadata=metadata,
        )
    raise ValueError("Legacy .pt bank must contain either 'arrays' or 'representations'")


def _metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    return str(value)
