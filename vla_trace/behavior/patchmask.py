"""PatchMask data generation for Stage 3 behavior probes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PATCHMASK_VARIANTS = frozenset(
    {"none", "mask_target", "mask_gripper", "mask_robot", "mask_robot_exc_gripper", "mask_background", "custom"}
)
PATCHMASK_MODES = frozenset({"none", "black", "background_fill", "mosaic"})


@dataclass(frozen=True)
class ImageMaskEvalConfig:
    """Runtime PatchMask config used by LIBERO-style evaluators."""

    variant: str
    mode: str
    mask_value: int = 0
    bg_ring_width: int = 8
    mosaic_block: int = 8

    def active(self) -> bool:
        return self.variant not in {"", "none"} and self.mode not in {"", "none"}


def apply_patch_mask(
    image: Any,
    mask: Any,
    *,
    mode: str,
    mask_value: int = 0,
    bg_ring_width: int = 8,
    mosaic_block: int = 8,
) -> np.ndarray:
    """Return an image with the selected region masked."""
    if mode not in PATCHMASK_MODES:
        raise ValueError(f"Unsupported patchmask mode: {mode}")
    output = np.asarray(image).copy()
    mask2d = _mask_bool_2d(mask, output.shape[:2])
    if mode == "none" or not mask2d.any():
        return output
    if mode == "black":
        output[mask2d] = np.asarray(mask_value, dtype=output.dtype)
        return output
    if mode == "background_fill":
        _fill_with_ring_background(output, mask2d, ring_width=bg_ring_width, fallback_value=mask_value)
        return output
    if mode == "mosaic":
        _mosaic_region(output, mask2d, block=mosaic_block)
        return output
    raise ValueError(f"Unsupported patchmask mode: {mode}")


def build_patch_mask_from_maps(
    masks: Mapping[str, Any],
    *,
    variant: str,
    instances: list[str] | None = None,
    categories: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Build a boolean mask for a PatchMask variant from named instance masks."""
    if variant not in PATCHMASK_VARIANTS:
        raise ValueError(f"Unsupported patchmask variant: {variant}")
    mask_items = {name: np.asarray(value).astype(bool) for name, value in masks.items()}
    if not mask_items:
        raise ValueError("PatchMask requires at least one instance mask")
    shape = next(iter(mask_items.values())).shape[:2]
    categories = categories or {}
    if variant == "none":
        return np.zeros(shape, dtype=bool)
    if variant == "custom":
        if not instances:
            raise ValueError("custom PatchMask requires --instance values")
        return _union_named(mask_items, instances)
    if variant == "mask_target":
        chosen = instances or [
            name
            for name in mask_items
            if categories.get(name, "").lower() in {"target", "object", "manipulated", "receptacle"}
        ]
        if not chosen:
            chosen = [name for name in mask_items if not _is_robot_like(name, categories.get(name, ""))]
        return _union_named(mask_items, chosen)
    if variant == "mask_gripper":
        return _union_matching(mask_items, categories, include=("gripper",))
    if variant == "mask_robot":
        return _union_matching(mask_items, categories, include=("robot", "arm", "base", "gripper"))
    if variant == "mask_robot_exc_gripper":
        robot = _union_matching(mask_items, categories, include=("robot", "arm", "base", "gripper"))
        gripper = _union_matching(mask_items, categories, include=("gripper",))
        return robot & ~gripper
    if variant == "mask_background":
        foreground = _union_named(mask_items, list(mask_items))
        return ~foreground
    raise ValueError(f"Unsupported patchmask variant: {variant}")


def detect_instance_seg_keys(obs: Mapping[str, Any]) -> tuple[str, str]:
    """Detect agent-view and wrist-view instance segmentation keys in an obs dict."""
    candidates = _integer_segmentation_candidates(obs)
    if len(candidates) < 2:
        raise RuntimeError(
            "PatchMask needs two integer instance-segmentation observations. "
            f"Found candidates: {candidates}"
        )
    lower = {key: key.lower() for key in candidates}
    agent_key = next((key for key in candidates if "agentview" in lower[key]), candidates[0])
    wrist_key = next(
        (
            key
            for key in candidates
            if "eye_in_hand" in lower[key] or "robot0_eye_in_hand" in lower[key]
        ),
        None,
    )
    if wrist_key is None:
        wrist_key = next((key for key in candidates if "hand" in lower[key] and "gripper" not in lower[key]), None)
    if wrist_key is None:
        wrist_key = candidates[1] if candidates[0] == agent_key else candidates[0]
    if wrist_key == agent_key:
        wrist_key = candidates[1] if candidates[0] == agent_key else candidates[0]
    return agent_key, wrist_key


def build_robot_target_masks(
    seg_agent: Any,
    seg_wrist: Any,
    env: Any,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build agent-view and wrist-view bool masks from LIBERO instance segmentations."""
    if variant not in PATCHMASK_VARIANTS - {"custom", "none"}:
        raise ValueError(f"Unsupported LIBERO PatchMask variant: {variant}")
    _ensure_segmentation_ids(env)
    seg_a_raw = np.asarray(seg_agent)
    seg_w_raw = np.asarray(seg_wrist)
    seg_inst_agent = env.get_segmentation_instances(np.array(seg_a_raw, copy=True))
    seg_inst_wrist = env.get_segmentation_instances(np.array(seg_w_raw, copy=True))

    if variant == "mask_robot":
        return _instance_positive(seg_inst_agent, "robot", seg_a_raw.shape[:2]), _instance_positive(
            seg_inst_wrist, "robot", seg_w_raw.shape[:2]
        )
    if variant == "mask_gripper":
        if "gripper" not in seg_inst_agent or "gripper" not in seg_inst_wrist:
            raise RuntimeError("PatchMask variant mask_gripper requires a 'gripper' instance mapping")
        return _instance_positive(seg_inst_agent, "gripper", seg_a_raw.shape[:2]), _instance_positive(
            seg_inst_wrist, "gripper", seg_w_raw.shape[:2]
        )
    if variant == "mask_robot_exc_gripper":
        robot_agent = _instance_positive(seg_inst_agent, "robot", seg_a_raw.shape[:2])
        robot_wrist = _instance_positive(seg_inst_wrist, "robot", seg_w_raw.shape[:2])
        gripper_id = getattr(env, "segmentation_gripper_id", None)
        if gripper_id is None:
            raise RuntimeError("PatchMask variant mask_robot_exc_gripper requires env.segmentation_gripper_id")
        gripper_value = int(gripper_id) + 1
        return robot_agent & (_mask_plane(seg_a_raw) != gripper_value), robot_wrist & (
            _mask_plane(seg_w_raw) != gripper_value
        )
    if variant == "mask_target":
        targets = list(getattr(env, "obj_of_interest", []) or [])
        return _union_instances(seg_inst_agent, targets, seg_a_raw.shape[:2]), _union_instances(
            seg_inst_wrist, targets, seg_w_raw.shape[:2]
        )
    if variant == "mask_background":
        foreground_agent = _union_instances(seg_inst_agent, list(seg_inst_agent), seg_a_raw.shape[:2])
        foreground_wrist = _union_instances(seg_inst_wrist, list(seg_inst_wrist), seg_w_raw.shape[:2])
        return ~foreground_agent, ~foreground_wrist
    raise ValueError(f"Unsupported LIBERO PatchMask variant: {variant}")


def apply_image_mask_to_obs_inplace(
    obs: dict[str, Any],
    env: Any,
    cfg: ImageMaskEvalConfig,
    seg_keys: tuple[str, str] | None = None,
    *,
    agent_image_key: str = "agentview_image",
    wrist_image_key: str = "robot0_eye_in_hand_image",
) -> dict[str, Any]:
    """Apply PatchMask to current LIBERO observation images before model inference."""
    if not cfg.active():
        return obs
    if seg_keys is None:
        seg_keys = detect_instance_seg_keys(obs)
    agent_seg_key, wrist_seg_key = seg_keys
    mask_agent, mask_wrist = build_robot_target_masks(obs[agent_seg_key], obs[wrist_seg_key], env, cfg.variant)
    if agent_image_key in obs:
        obs[agent_image_key] = apply_patch_mask(
            np.ascontiguousarray(obs[agent_image_key]),
            mask_agent,
            mode=cfg.mode,
            mask_value=cfg.mask_value,
            bg_ring_width=cfg.bg_ring_width,
            mosaic_block=cfg.mosaic_block,
        )
    if wrist_image_key in obs:
        obs[wrist_image_key] = apply_patch_mask(
            np.ascontiguousarray(obs[wrist_image_key]),
            mask_wrist,
            mode=cfg.mode,
            mask_value=cfg.mask_value,
            bg_ring_width=cfg.bg_ring_width,
            mosaic_block=cfg.mosaic_block,
        )
    return obs


def apply_patch_mask_to_views(
    images: Mapping[str, Any],
    masks_by_view: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
    mode: str,
    instances: list[str] | None = None,
    categories: Mapping[str, str] | None = None,
    mask_value: int = 0,
    bg_ring_width: int = 8,
    mosaic_block: int = 8,
) -> dict[str, np.ndarray]:
    """Apply one PatchMask setting to multiple camera views offline."""
    outputs: dict[str, np.ndarray] = {}
    for view, image in images.items():
        if view not in masks_by_view:
            continue
        selected = build_patch_mask_from_maps(
            masks_by_view[view],
            variant=variant,
            instances=instances,
            categories=categories,
        )
        outputs[view] = apply_patch_mask(
            image,
            selected,
            mode=mode,
            mask_value=mask_value,
            bg_ring_width=bg_ring_width,
            mosaic_block=mosaic_block,
        )
    return outputs


def write_patchmask_artifact(
    image_path: str | Path,
    mask_path: str | Path,
    output_image: str | Path,
    *,
    variant: str,
    mode: str,
    instances: list[str] | None = None,
    categories: Mapping[str, str] | None = None,
    output_manifest: str | Path | None = None,
    image_key: str | None = None,
    mask_keys: list[str] | None = None,
    mask_value: int = 0,
    bg_ring_width: int = 8,
    mosaic_block: int = 8,
) -> dict[str, Any]:
    image = _load_array(image_path, key=image_key)
    masks = _load_mask_map(mask_path, keys=mask_keys)
    selected_mask = build_patch_mask_from_maps(masks, variant=variant, instances=instances, categories=categories)
    masked = apply_patch_mask(
        image,
        selected_mask,
        mode=mode,
        mask_value=mask_value,
        bg_ring_width=bg_ring_width,
        mosaic_block=mosaic_block,
    )
    target = Path(output_image)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, masked)
    manifest = {
        "variant": variant,
        "mode": mode,
        "instances": instances or [],
        "mask_source": str(mask_path),
        "image_source": str(image_path),
        "output_image": str(target),
        "selected_pixels": int(selected_mask.sum()),
        "params": {
            "mask_value": mask_value,
            "bg_ring_width": bg_ring_width,
            "mosaic_block": mosaic_block,
        },
    }
    if output_manifest:
        _write_json(output_manifest, manifest)
    return manifest


def _mask_bool_2d(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != shape:
        rows = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(int)
        cols = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(int)
        array = array[np.ix_(rows, cols)]
    return array.astype(bool)


def _mask_plane(array: Any) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0]
    return values


def _integer_segmentation_candidates(obs: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for key, value in obs.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim not in (2, 3):
            continue
        if value.ndim == 3 and value.shape[-1] != 1:
            continue
        if np.issubdtype(value.dtype, np.integer):
            keys.append(str(key))
    return keys


def _ensure_segmentation_ids(env: Any) -> None:
    if getattr(env, "segmentation_robot_id", None) is not None and getattr(env, "segmentation_gripper_id", None) is not None:
        return
    try:
        instances = list(env.env.model.instances_to_ids.keys())
    except AttributeError:
        return
    robot_keywords = ("Panda", "Robot", "robot", "UR5", "IIWA", "Sawyer", "Jaco")
    gripper_keywords = ("Gripper", "gripper")
    if getattr(env, "segmentation_robot_id", None) is None:
        for idx, name in enumerate(instances):
            if any(piece in name for piece in robot_keywords):
                env.segmentation_robot_id = idx
                break
        if getattr(env, "segmentation_robot_id", None) is None and instances:
            env.segmentation_robot_id = 0
    if getattr(env, "segmentation_gripper_id", None) is None:
        for idx, name in enumerate(instances):
            if any(piece in name for piece in gripper_keywords):
                env.segmentation_gripper_id = idx
                break
    if not getattr(env, "segmentation_id_mapping", None) and instances:
        robot_names = {"Panda0", "PandaGripper0", "RethinkMount0"}
        robot_id = getattr(env, "segmentation_robot_id", 0)
        env.segmentation_id_mapping = {
            idx: name
            for idx, name in enumerate(instances)
            if name not in robot_names and idx != robot_id
        }
        env.instance_to_id = {name: idx + 1 for idx, name in env.segmentation_id_mapping.items()}


def _instance_positive(instances: Mapping[str, Any], name: str, shape: tuple[int, int]) -> np.ndarray:
    if name not in instances:
        return np.zeros(shape, dtype=bool)
    return _mask_plane(instances[name]) > 0


def _union_instances(instances: Mapping[str, Any], names: list[str], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for name in names:
        if name in instances:
            mask |= _mask_plane(instances[name]) > 0
    return mask


def _fill_with_ring_background(image: np.ndarray, mask: np.ndarray, *, ring_width: int, fallback_value: int) -> None:
    ring = _dilate_mask(mask, int(ring_width)) & ~mask
    source = ring if ring.any() else ~mask
    if source.any():
        fill = image[source].reshape(-1, image.shape[-1]).mean(axis=0)
        if image.dtype == np.uint8:
            fill = np.clip(np.round(fill), 0, 255).astype(np.uint8)
        else:
            fill = fill.astype(image.dtype)
        image[mask] = fill
    else:
        image[mask] = np.asarray(fallback_value, dtype=image.dtype)


def _mosaic_region(image: np.ndarray, mask: np.ndarray, *, block: int) -> None:
    block = max(int(block), 1)
    height, width, channels = image.shape
    for y0 in range(0, height, block):
        y1 = min(y0 + block, height)
        for x0 in range(0, width, block):
            x1 = min(x0 + block, width)
            submask = mask[y0:y1, x0:x1]
            if not submask.any():
                continue
            patch = image[y0:y1, x0:x1].copy()
            fill = patch.reshape(-1, channels).mean(axis=0)
            if image.dtype == np.uint8:
                fill = np.clip(np.round(fill), 0, 255).astype(np.uint8)
            else:
                fill = fill.astype(image.dtype)
            patch[submask] = fill
            image[y0:y1, x0:x1] = patch


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    side = 2 * radius + 1
    for dy in range(side):
        for dx in range(side):
            out |= padded[dy : dy + height, dx : dx + width]
    return out


def _union_named(masks: Mapping[str, np.ndarray], names: list[str]) -> np.ndarray:
    shape = next(iter(masks.values())).shape[:2]
    out = np.zeros(shape, dtype=bool)
    for name in names:
        if name in masks:
            out |= masks[name].astype(bool)
    return out


def _union_matching(masks: Mapping[str, np.ndarray], categories: Mapping[str, str], *, include: tuple[str, ...]) -> np.ndarray:
    chosen = []
    for name in masks:
        text = f"{name} {categories.get(name, '')}".lower()
        if any(piece in text for piece in include):
            chosen.append(name)
    return _union_named(masks, chosen)


def _is_robot_like(name: str, category: str) -> bool:
    text = f"{name} {category}".lower()
    return any(piece in text for piece in ("robot", "gripper", "arm", "base", "table"))


def _load_array(path: str | Path, key: str | None) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            chosen = key or payload.files[0]
            return np.asarray(payload[chosen])
    raise ValueError(f"Unsupported array format: {source.suffix}")


def _load_mask_map(path: str | Path, keys: list[str] | None) -> dict[str, np.ndarray]:
    source = Path(path)
    if source.suffix == ".npy":
        return {(keys or ["mask"])[0]: np.asarray(np.load(source, allow_pickle=False))}
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            names = keys or payload.files
            return {name: np.asarray(payload[name]) for name in names}
    raise ValueError(f"Unsupported mask format: {source.suffix}")


def _write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")
    return target
