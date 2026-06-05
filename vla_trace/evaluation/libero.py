"""Path-clean LIBERO rollout runner used by the public CLI."""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vla_trace.behavior.patchmask import (
    PATCHMASK_MODES,
    PATCHMASK_VARIANTS,
    ImageMaskEvalConfig,
    apply_image_mask_to_obs_inplace,
    detect_instance_seg_keys,
)
from vla_trace.evaluation.adapters import LiberoStep, PolicyBuildRequest, build_policy
from vla_trace.io import load_config, write_json

LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
}


@dataclass(frozen=True)
class LiberoEvalRequest:
    """Resolved evaluation request for real, mock, or dry-run LIBERO jobs."""

    model: str
    dataset: str
    output_dir: str
    model_path: str | None = None
    data_root: str | None = None
    model_config: str | None = None
    benchmark_config: str | None = None
    config_path: str | None = None
    adapter_factory: str | None = None
    vlm4vla_root: str | None = None
    openpi_root: str | None = None
    openpi_config_name: str | None = None
    tokenizer_path: str | None = None
    device: str = "cuda"
    seed: int = 0
    task_ids: tuple[int, ...] = ()
    num_trials_per_task: int = 20
    max_steps: int | None = None
    num_steps_wait: int = 10
    execute_step: int = 1
    replan_steps: int | None = None
    center_crop: bool = False
    save_video: bool = False
    video_every: int = 0
    use_openvla_prompt: bool = False
    single_unnorm: bool = False
    knockout_config: dict[str, Any] | None = None
    patchmask_config: dict[str, Any] | None = None
    setting: str = "baseline"
    result_name: str | None = None
    mock_env: bool = False
    dry_run: bool = False
    extra_config: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_task_ids(self) -> tuple[int, ...]:
        if self.task_ids:
            return self.task_ids
        return tuple(range(SUITE_TASK_COUNTS.get(self.dataset, 10)))

    @property
    def resolved_max_steps(self) -> int:
        return int(self.max_steps or LIBERO_MAX_STEPS.get(self.dataset, 400))

    @property
    def test_num(self) -> int:
        return len(self.resolved_task_ids) * int(self.num_trials_per_task)


def build_libero_eval_plan(request: LiberoEvalRequest) -> dict[str, Any]:
    """Build a serializable plan without importing simulator/model packages."""

    task_ids = list(request.resolved_task_ids)
    jobs = []
    for task_id in task_ids:
        for episode_id in range(request.num_trials_per_task):
            jobs.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "max_steps": request.resolved_max_steps,
                    "num_steps_wait": request.num_steps_wait,
                    "save_video": _should_save_video(
                        episode_id,
                        request.num_trials_per_task,
                        request.video_every,
                        enabled=request.save_video,
                    ),
                }
            )
    return {
        "status": "planned",
        "command": "eval-libero",
        "model": request.model,
        "dataset": request.dataset,
        "suite": request.dataset,
        "setting": request.setting,
        "task_ids": task_ids,
        "num_trials_per_task": request.num_trials_per_task,
        "test_num": request.test_num,
        "max_steps": request.resolved_max_steps,
        "output_dir": request.output_dir,
        "model_path": request.model_path,
        "data_root": request.data_root,
        "adapter_factory": request.adapter_factory,
        "vlm4vla_root": request.vlm4vla_root,
        "openpi_config_name": request.openpi_config_name,
        "knockout_config": request.knockout_config,
        "patchmask_config": request.patchmask_config,
        "requires_instance_segmentation": bool(_patchmask_runtime_config(request.patchmask_config)),
        "jobs": jobs,
    }


def run_libero_evaluation(request: LiberoEvalRequest) -> dict[str, Any]:
    """Run LIBERO online evaluation and write a plot-compatible result JSON."""

    output_root = Path(request.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = build_libero_eval_plan(request)
    if request.dry_run:
        path = output_root / (request.result_name or f"{request.dataset}_{request.setting}_plan.json")
        write_json(path, plan)
        return {**plan, "result_path": str(path)}

    if request.mock_env:
        result = _run_mock_eval(request, plan)
    else:
        result = _run_real_libero_eval(request, plan)
    result_path = _write_eval_result(request, result)
    return {**result, "result_path": str(result_path)}


def _run_mock_eval(request: LiberoEvalRequest, plan: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(request.seed)
    trials = []
    successes = 0
    for job in plan["jobs"]:
        success = bool(rng.random() >= 0.25)
        successes += int(success)
        trials.append(
            {
                **job,
                "success": success,
                "steps": min(3, request.resolved_max_steps),
                "task_description": f"mock {request.dataset} task {job['task_id']}",
            }
        )
    return _result_payload(request, successes, request.test_num, trials, status="ok", backend="mock")


def _run_real_libero_eval(request: LiberoEvalRequest, plan: dict[str, Any]) -> dict[str, Any]:
    _seed_everything(request.seed)
    _prepare_libero_runtime(request)
    policy = build_policy(
        PolicyBuildRequest(
            model=request.model,
            dataset=request.dataset,
            model_path=request.model_path,
            data_root=request.data_root,
            output_dir=request.output_dir,
            model_config=request.model_config,
            benchmark_config=request.benchmark_config,
            config_path=request.config_path,
            device=request.device,
            seed=request.seed,
            center_crop=request.center_crop,
            execute_step=request.execute_step,
            replan_steps=request.replan_steps,
            openpi_config_name=request.openpi_config_name,
            openpi_root=request.openpi_root,
            tokenizer_path=request.tokenizer_path,
            vlm4vla_root=request.vlm4vla_root,
            use_openvla_prompt=request.use_openvla_prompt,
            single_unnorm=request.single_unnorm,
            knockout_config=request.knockout_config,
            extra_config=request.extra_config,
        ),
        adapter_factory=request.adapter_factory,
    )
    policy.configure_knockout(request.knockout_config)

    benchmark_module, OffScreenRenderEnv, SegmentationRenderEnv, get_libero_path = _import_libero_stack()
    benchmark_dict = benchmark_module.get_benchmark_dict()
    if request.dataset not in benchmark_dict:
        raise ValueError(f"LIBERO suite {request.dataset!r} not found in local LIBERO installation")
    task_suite = benchmark_dict[request.dataset]()

    trials: list[dict[str, Any]] = []
    successes = 0
    for task_id in request.resolved_task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _make_libero_env(
            task,
            OffScreenRenderEnv,
            SegmentationRenderEnv,
            get_libero_path,
            seed=request.seed,
            use_instance_segmentation=bool(_patchmask_runtime_config(request.patchmask_config)),
        )
        try:
            for episode_id in range(request.num_trials_per_task):
                if episode_id >= len(initial_states):
                    raise ValueError(
                        f"Requested episode {episode_id} for task {task_id}, "
                        f"but local LIBERO only exposes {len(initial_states)} initial states"
                    )
                env.reset()
                obs = env.set_init_state(initial_states[episode_id])
                save_video = _should_save_video(
                    episode_id,
                    request.num_trials_per_task,
                    request.video_every,
                    enabled=request.save_video,
                )
                success, record = _run_single_episode(
                    request,
                    policy,
                    env,
                    obs,
                    task_description,
                    task_id=task_id,
                    episode_id=episode_id,
                    save_video=save_video,
                )
                successes += int(success)
                trials.append(record)
        finally:
            if hasattr(env, "close"):
                env.close()
    return _result_payload(request, successes, request.test_num, trials, status="ok", backend="libero")


def _run_single_episode(
    request: LiberoEvalRequest,
    policy: Any,
    env: Any,
    obs: Mapping[str, Any],
    task_description: str,
    *,
    task_id: int,
    episode_id: int,
    save_video: bool,
) -> tuple[bool, dict[str, Any]]:
    policy.reset()
    images: list[np.ndarray] = []
    actions: list[list[float]] = []
    success = False
    steps_taken = 0
    wait_action = get_libero_dummy_action(request.model)
    patchmask_cfg = _patchmask_runtime_config(request.patchmask_config)
    patchmask_seg_keys: tuple[str, str] | None = None
    for _ in range(request.num_steps_wait):
        obs, _reward, done, _info = env.step(wait_action)
        if done:
            success = True
            break
    if not success:
        for step_id in range(request.resolved_max_steps):
            obs_for_policy = obs
            if patchmask_cfg is not None:
                if not isinstance(obs_for_policy, dict):
                    obs_for_policy = dict(obs_for_policy)
                try:
                    if patchmask_seg_keys is None:
                        patchmask_seg_keys = detect_instance_seg_keys(obs_for_policy)
                    apply_image_mask_to_obs_inplace(obs_for_policy, env, patchmask_cfg, seg_keys=patchmask_seg_keys)
                except Exception as exc:
                    raise RuntimeError(
                        "PatchMask online evaluation requires LIBERO observations with instance segmentation. "
                        "Make sure eval-libero created SegmentationRenderEnv(camera_segmentations='instance') "
                        "and your local LIBERO installation exposes agentview and wrist segmentation keys."
                    ) from exc
            step = build_libero_step(
                obs_for_policy,
                task_description,
                family=request.model,
                task_id=task_id,
                episode_id=episode_id,
                step_id=step_id,
            )
            images.append(step.image)
            action = policy.predict_action(step)
            action_list = action.astype(float).tolist()
            actions.append(action_list)
            obs, _reward, done, _info = env.step(action_list)
            steps_taken = step_id + 1
            if done:
                success = True
                break
    if save_video:
        _save_episode_artifacts(request, task_id, episode_id, success, images, actions)
    return success, {
        "task_id": task_id,
        "episode_id": episode_id,
        "task_description": task_description,
        "success": success,
        "steps": steps_taken,
        "save_video": save_video,
    }


def build_libero_step(
    obs: Mapping[str, Any],
    task_description: str,
    *,
    family: str,
    task_id: int,
    episode_id: int,
    step_id: int,
) -> LiberoStep:
    if family == "pi05":
        image, wrist_image = get_libero_image_pi0(obs, 224)
        state = get_libero_state(obs)
    else:
        image = get_libero_image(obs, 224)
        wrist_image = None
        state = None
    return LiberoStep(
        raw_observation=obs,
        image=image,
        wrist_image=wrist_image,
        state=state,
        task_description=task_description,
        task_id=task_id,
        episode_id=episode_id,
        step_id=step_id,
    )


def get_libero_image(obs: Mapping[str, Any], resize_size: int | tuple[int, int] = 224) -> np.ndarray:
    """OpenVLA-style LIBERO image preprocessing from the VLM4VLA runner."""

    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = np.asarray(obs["agentview_image"])[::-1, ::-1]
    try:
        import tensorflow as tf

        encoded = tf.image.encode_jpeg(img)
        decoded = tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8)
        resized = tf.image.resize(decoded, resize_size, method="lanczos3", antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()
    except Exception:
        from PIL import Image as PILImage
        import io

        pil_img = PILImage.fromarray(np.ascontiguousarray(img))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        pil_img = PILImage.open(buffer)
        pil_img.load()
        pil_img = pil_img.resize((resize_size[1], resize_size[0]), resample=PILImage.LANCZOS)
        return np.asarray(pil_img, dtype=np.uint8)


def get_libero_image_pi0(obs: Mapping[str, Any], resize_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """pi0.5/OpenPI LIBERO image preprocessing from the VLM4VLA runner."""

    img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])
    wrist_img = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1])
    return (
        _resize_with_pad_pil(img, resize_size, resize_size).astype(np.uint8),
        _resize_with_pad_pil(wrist_img, resize_size, resize_size).astype(np.uint8),
    )


def get_libero_state(obs: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )


def get_libero_dummy_action(_model_family: str = "openvla") -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def resolve_knockout_config_for_eval(
    *,
    model: str,
    phase: str,
    mode: str,
    layers: Sequence[int] | str | None,
    direction: str | None,
    text_scope: str,
    prefill_mode: str | None = None,
    generation_mode: str | None = None,
) -> dict[str, Any] | None:
    """Build the compact knockout dict expected by VLM4VLA-style adapters."""

    if mode == "baseline" and direction is None and not prefill_mode and not generation_mode:
        return None
    ko_layers: Sequence[int] | str
    if layers is None:
        ko_layers = "all"
    elif isinstance(layers, str):
        ko_layers = layers
    else:
        ko_layers = [int(layer) for layer in layers]
    config = {
        "mode": mode,
        "knockout_layers": ko_layers,
        "direction": direction,
        "knockout_phase": phase,
        "text_knockout_scope": text_scope,
        "family": model,
    }
    if prefill_mode or generation_mode:
        config["knockout_phase"] = "both"
        config["phase_policy"] = {
            "prefill": {"mode": prefill_mode} if prefill_mode else {},
            "generation": {"mode": generation_mode} if generation_mode else {},
        }
    return config


def resolve_patchmask_config_for_eval(
    *,
    variant: str | None,
    mode: str | None,
    mask_value: int = 0,
    bg_ring_width: int = 8,
    mosaic_block: int = 8,
) -> dict[str, Any] | None:
    """Build the compact PatchMask dict used by LIBERO online evaluation."""

    variant_name = str(variant or "none")
    mode_name = str(mode or "none")
    if variant_name not in PATCHMASK_VARIANTS - {"custom"}:
        raise ValueError(f"Unsupported LIBERO PatchMask variant: {variant_name}")
    if mode_name not in PATCHMASK_MODES:
        raise ValueError(f"Unsupported PatchMask mode: {mode_name}")
    if variant_name == "none" and mode_name == "none":
        return None
    if variant_name == "none" or mode_name == "none":
        raise ValueError("--patchmask-variant and --patchmask-mode must both be non-none, or both be none")
    return {
        "variant": variant_name,
        "mode": mode_name,
        "mask_value": int(mask_value),
        "bg_ring_width": int(bg_ring_width),
        "mosaic_block": int(mosaic_block),
    }


def setting_name_from_patchmask(patchmask_config: dict[str, Any] | None) -> str:
    if not patchmask_config:
        return "baseline"
    return f"patchmask_{patchmask_config['variant']}_{patchmask_config['mode']}"


def setting_name_from_knockout(knockout_config: dict[str, Any] | None) -> str:
    if not knockout_config:
        return "baseline"
    phase = str(knockout_config.get("knockout_phase") or knockout_config.get("phase") or "generation")
    mode = str(knockout_config.get("mode") or "baseline")
    if mode == "baseline":
        return "baseline"
    if "__" in mode:
        parts = mode.split("_", 1)
        return parts[1] if len(parts) == 2 and parts[0] in {"pi05", "openvla"} else mode
    return f"{phase}_{mode}"


def _result_payload(
    request: LiberoEvalRequest,
    success_num: int,
    test_num: int,
    trials: list[dict[str, Any]],
    *,
    status: str,
    backend: str,
) -> dict[str, Any]:
    rate = float(success_num / test_num) if test_num else 0.0
    ci_low, ci_high = _wilson_interval(success_num, test_num)
    payload = {
        "status": status,
        "command": "eval-libero",
        "backend": backend,
        "model": request.model,
        "dataset": request.dataset,
        "suite": request.dataset,
        "setting": request.setting,
        "success_rate": rate,
        "success_rate_percent": rate * 100.0,
        "success_num": int(success_num),
        "test_num": int(test_num),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "task_ids": list(request.resolved_task_ids),
        "num_trials_per_task": request.num_trials_per_task,
        "max_steps": request.resolved_max_steps,
        "model_path": request.model_path,
        "data_root": request.data_root,
        "knockout_config": request.knockout_config,
        "patchmask_config": request.patchmask_config,
        "trials": trials,
        "created_at_unix": int(time.time()),
    }
    _attach_layer_metadata(payload, request.knockout_config)
    return payload


def _patchmask_runtime_config(config: dict[str, Any] | None) -> ImageMaskEvalConfig | None:
    if not config:
        return None
    cfg = ImageMaskEvalConfig(
        variant=str(config.get("variant", "none")),
        mode=str(config.get("mode", "none")),
        mask_value=int(config.get("mask_value", 0)),
        bg_ring_width=int(config.get("bg_ring_width", 8)),
        mosaic_block=int(config.get("mosaic_block", 8)),
    )
    return cfg if cfg.active() else None


def _attach_layer_metadata(payload: dict[str, Any], knockout_config: dict[str, Any] | None) -> None:
    if not knockout_config:
        return
    layers = knockout_config.get("knockout_layers")
    if isinstance(layers, list) and len(layers) == 1:
        payload["layer"] = int(layers[0])
    elif isinstance(layers, list) and layers:
        payload["layers"] = [int(layer) for layer in layers]
    if isinstance(layers, str):
        payload["layers"] = layers
    if "center_layer" in knockout_config:
        payload["layer"] = int(knockout_config["center_layer"])
    if "window_size" in knockout_config:
        payload["window"] = int(knockout_config["window_size"])


def _write_eval_result(request: LiberoEvalRequest, payload: dict[str, Any]) -> Path:
    name = request.result_name or _default_result_name(request, payload)
    if not name.endswith(".json"):
        name = f"{name}.json"
    return write_json(Path(request.output_dir) / name, payload)


def _default_result_name(request: LiberoEvalRequest, payload: dict[str, Any]) -> str:
    layer = payload.get("layer")
    window = payload.get("window")
    if layer is not None and window is not None:
        suffix = f"layer{layer}_w{window}"
    elif layer is not None:
        suffix = f"layer{layer}"
    else:
        suffix = request.setting
    return f"{request.dataset}_{suffix}_{payload['success_rate']:.4f}.json"


def _import_libero_stack() -> tuple[Any, Any, Any, Any]:
    try:
        from libero import benchmark, get_libero_path
        from libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "Real LIBERO evaluation requires a local LIBERO installation. "
            "Use --dry-run to validate commands or --mock-env for CI smoke tests."
        ) from exc
    try:
        from libero.envs import SegmentationRenderEnv
    except ImportError:
        SegmentationRenderEnv = None
    return benchmark, OffScreenRenderEnv, SegmentationRenderEnv, get_libero_path


def _prepare_libero_runtime(request: LiberoEvalRequest) -> None:
    """Prepare non-interactive LIBERO import/config behavior."""

    if request.vlm4vla_root:
        local_parent = Path(request.vlm4vla_root).expanduser() / "eval" / "libero"
        if local_parent.is_dir() and str(local_parent) not in sys.path:
            sys.path.insert(0, str(local_parent))
    if not request.data_root:
        return
    config_root = os.environ.get("LIBERO_CONFIG_PATH")
    if config_root and (Path(config_root) / "config.yaml").exists():
        return

    import importlib.util
    import yaml

    data_root = Path(request.data_root).expanduser()
    benchmark_root = data_root if (data_root / "bddl_files").is_dir() else None
    spec = importlib.util.find_spec("libero")
    if benchmark_root is None and spec and spec.origin:
        benchmark_root = Path(spec.origin).resolve().parent
    if benchmark_root is None:
        return

    config_dir = Path(request.output_dir) / ".libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_path = config_dir / "config.yaml"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(data_root),
        "assets": str(benchmark_root / "assets"),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def _make_libero_env(
    task: Any,
    OffScreenRenderEnv: Any,
    SegmentationRenderEnv: Any,
    get_libero_path: Any,
    *,
    seed: int,
    use_instance_segmentation: bool = False,
) -> tuple[Any, str]:
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_kwargs = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    if use_instance_segmentation:
        if SegmentationRenderEnv is not None:
            env = SegmentationRenderEnv(**env_kwargs, camera_segmentations="instance")
        else:
            try:
                env = OffScreenRenderEnv(**env_kwargs, camera_segmentations="instance")
            except TypeError as exc:
                raise RuntimeError(
                    "PatchMask online evaluation requires a LIBERO SegmentationRenderEnv "
                    "or an OffScreenRenderEnv that accepts camera_segmentations='instance'."
                ) from exc
    else:
        env = OffScreenRenderEnv(**env_kwargs)
    env.seed(seed)
    return env, task_description


def _resize_with_pad_pil(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(image)
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.asarray(image)
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=PILImage.BILINEAR)
    zero_image = PILImage.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.asarray(zero_image)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator).astype(np.float32)


def _should_save_video(episode_id: int, num_trials: int, video_every: int, *, enabled: bool) -> bool:
    if not enabled:
        return False
    if video_every <= 0:
        return episode_id in set(np.linspace(0, num_trials - 1, min(5, num_trials), dtype=int))
    return episode_id % video_every == 0 or episode_id in set(np.linspace(0, num_trials - 1, min(5, num_trials), dtype=int))


def _save_episode_artifacts(
    request: LiberoEvalRequest,
    task_id: int,
    episode_id: int,
    success: bool,
    images: Sequence[np.ndarray],
    actions: Sequence[Sequence[float]],
) -> None:
    if not images:
        return
    videos_dir = Path(request.output_dir) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    status = "success" if success else "failure"
    video_path = videos_dir / f"{status}_task{task_id}_episode{episode_id}.mp4"
    try:
        import imageio

        with imageio.get_writer(video_path, fps=40, format="FFMPEG", mode="I") as writer:
            for image in images:
                writer.append_data(np.asarray(image, dtype=np.uint8))
    except Exception:
        frames_path = videos_dir / f"{status}_task{task_id}_episode{episode_id}_frames.npz"
        np.savez_compressed(frames_path, images=np.asarray(images, dtype=np.uint8), actions=np.asarray(actions))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _wilson_interval(success_num: int, test_num: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if test_num <= 0:
        return None, None
    p = success_num / test_num
    denom = 1 + z**2 / test_num
    centre = p + z**2 / (2 * test_num)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * test_num)) / test_num)
    return (centre - spread) / denom, (centre + spread) / denom


def request_from_config(config: str | None, overrides: Mapping[str, Any]) -> LiberoEvalRequest:
    """Merge an optional YAML/JSON eval config with CLI overrides."""

    cfg = load_config(config) if config else {}
    merged = {**cfg, **{key: value for key, value in overrides.items() if value is not None}}
    if "task_ids" in merged and isinstance(merged["task_ids"], str):
        merged["task_ids"] = tuple(_parse_int_csv_or_all(merged["task_ids"]))
    elif "task_ids" in merged and merged["task_ids"] in (None, "all"):
        merged["task_ids"] = ()
    elif "task_ids" in merged:
        merged["task_ids"] = tuple(int(item) for item in merged["task_ids"])
    return LiberoEvalRequest(**merged)


def _parse_int_csv_or_all(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return ()
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    return tuple(int(piece) for piece in pieces)
