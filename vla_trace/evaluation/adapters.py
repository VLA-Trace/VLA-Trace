"""Policy adapter boundary for online LIBERO evaluation.

The public toolkit keeps heavy model execution behind a small adapter contract.
Users can pass their own factory, use a lightweight OpenVLA HuggingFace adapter,
or point to a local VLM4VLA checkout for compatibility with existing training
configs.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from vla_trace.io import load_config


class AdapterError(RuntimeError):
    """Raised when a requested policy adapter cannot be constructed."""


@dataclass(frozen=True)
class LiberoStep:
    """Preprocessed LIBERO observation passed to policy adapters."""

    raw_observation: Mapping[str, Any]
    image: np.ndarray
    wrist_image: np.ndarray | None
    state: np.ndarray | None
    task_description: str
    task_id: int
    episode_id: int
    step_id: int


@dataclass(frozen=True)
class PolicyBuildRequest:
    """Everything a policy factory needs without relying on local paths."""

    model: str
    dataset: str
    model_path: str | None = None
    data_root: str | None = None
    output_dir: str | None = None
    model_config: str | None = None
    benchmark_config: str | None = None
    config_path: str | None = None
    device: str = "cuda"
    seed: int = 0
    center_crop: bool = False
    execute_step: int = 1
    replan_steps: int | None = None
    openpi_config_name: str | None = None
    openpi_root: str | None = None
    tokenizer_path: str | None = None
    vlm4vla_root: str | None = None
    unnorm_key: str | None = None
    use_openvla_prompt: bool = False
    single_unnorm: bool = False
    knockout_config: dict[str, Any] | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RolloutPolicy(Protocol):
    """Minimal callable policy interface."""

    def reset(self) -> None:
        ...

    def predict_action(self, step: LiberoStep) -> Sequence[float] | np.ndarray:
        ...


PolicyFactory = Callable[[PolicyBuildRequest], Any]


class PolicyRuntimeAdapter:
    """Normalize common policy method names into one rollout contract."""

    def __init__(self, policy: Any, *, family: str, knockout_configured: bool = False) -> None:
        self.policy = policy
        self.family = family
        self.knockout_configured = knockout_configured

    @property
    def is_pi0(self) -> bool:
        return bool(getattr(self.policy, "is_pi0", False)) or self.family == "pi05"

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def configure_knockout(self, knockout_config: dict[str, Any] | None) -> None:
        """Forward knockout updates when a custom adapter supports it."""
        if hasattr(self.policy, "configure_knockout"):
            self.policy.configure_knockout(knockout_config)
        elif hasattr(self.policy, "set_knockout_config"):
            self.policy.set_knockout_config(knockout_config)
        elif knockout_config and not self._has_static_knockout_config():
            raise AdapterError(
                "This policy adapter does not expose a knockout hook. Use a "
                "VLM4VLA-compatible adapter or implement configure_knockout(config) "
                "on your custom policy before running Stage 2 interventions."
            )

    def _has_static_knockout_config(self) -> bool:
        if self.knockout_configured:
            return True
        configs = getattr(self.policy, "configs", None)
        return isinstance(configs, dict) and bool(configs.get("knockout_config"))

    def predict_action(self, step: LiberoStep) -> np.ndarray:
        policy = self.policy
        if hasattr(policy, "predict_action"):
            action = _call_predict_action(policy.predict_action, step)
        elif self.is_pi0 and hasattr(policy, "step_pi0"):
            if step.wrist_image is None or step.state is None:
                raise AdapterError("pi0.5 policy requires wrist_image and state in the LIBERO observation")
            action = policy.step_pi0(step.image, step.wrist_image, step.state, step.task_description)
        elif hasattr(policy, "step"):
            action = policy.step(step.image, step.task_description)
        else:
            raise AdapterError(
                "Policy must implement predict_action(step), step(image, task), "
                "or step_pi0(image, wrist_image, state, task)."
            )
        return _to_action_array(action)


def load_policy_factory(import_path: str) -> PolicyFactory:
    """Load `module:function` or `module.function` policy factories."""

    if ":" in import_path:
        module_name, attr_name = import_path.split(":", 1)
    else:
        module_name, attr_name = import_path.rsplit(".", 1)
    if not module_name or not attr_name:
        raise AdapterError(f"Invalid adapter factory path: {import_path!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise AdapterError(f"Adapter factory is not callable: {import_path}")
    return factory


def build_policy(request: PolicyBuildRequest, *, adapter_factory: str | None = None) -> PolicyRuntimeAdapter:
    """Build a rollout policy from a custom or built-in adapter."""

    if adapter_factory:
        factory = load_policy_factory(adapter_factory)
        policy = _call_factory(factory, request)
        if isinstance(policy, PolicyRuntimeAdapter):
            return policy
        return PolicyRuntimeAdapter(policy, family=request.model, knockout_configured=bool(request.knockout_config))

    if request.vlm4vla_root:
        return PolicyRuntimeAdapter(
            _build_vlm4vla_policy(request),
            family=request.model,
            knockout_configured=bool(request.knockout_config),
        )

    if request.model == "openvla":
        return PolicyRuntimeAdapter(OpenVLAHFPolicy(request), family=request.model)

    raise AdapterError(
        "pi0.5 online inference needs a model-specific adapter. Pass "
        "--adapter-factory your_pkg:create_policy, or use --vlm4vla-root with "
        "a VLM4VLA/OpenPI-compatible config."
    )


def _call_factory(factory: PolicyFactory, request: PolicyBuildRequest) -> Any:
    try:
        return factory(request)
    except TypeError as exc:
        signature = inspect.signature(factory)
        if len(signature.parameters) == 0:
            return factory()
        try:
            return factory(**request.to_dict())
        except TypeError:
            raise exc


def _call_predict_action(func: Callable[..., Any], step: LiberoStep) -> Any:
    try:
        return func(step)
    except TypeError as exc:
        signature = inspect.signature(func)
        parameter_count = len(signature.parameters)
        if parameter_count == 2:
            return func(step.raw_observation, step.task_description)
        if parameter_count >= 4 and step.wrist_image is not None and step.state is not None:
            return func(step.image, step.wrist_image, step.state, step.task_description)
        if parameter_count >= 2:
            return func(step.image, step.task_description)
        raise exc


def _to_action_array(action: Any) -> np.ndarray:
    if isinstance(action, tuple) and action:
        action = action[0]
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 0:
        raise AdapterError("Policy returned a scalar action; expected a LIBERO action vector")
    if array.ndim > 1:
        array = array.reshape(-1, array.shape[-1])[0]
    return array


class OpenVLAHFPolicy:
    """Small OpenVLA HuggingFace adapter for checkpoints with `predict_action`."""

    def __init__(self, request: PolicyBuildRequest) -> None:
        if not request.model_path:
            raise AdapterError("OpenVLA HuggingFace adapter requires --model-path")
        self.request = request
        self.model_path = request.model_path
        self.device = request.device
        self.unnorm_key = request.unnorm_key or f"{request.dataset}_no_noops"
        self.use_openvla_prompt = request.use_openvla_prompt
        self._loaded = False
        self._torch = None
        self._processor = None
        self._model = None
        self.load()

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise AdapterError(
                "OpenVLA HuggingFace inference requires optional dependencies. "
                "Install with `pip install -e .[openvla]` and make sure the "
                "checkpoint supports trust_remote_code predict_action()."
            ) from exc
        self._torch = torch
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    def reset(self) -> None:
        return None

    def predict_action(self, step: LiberoStep) -> np.ndarray:
        if self._processor is None or self._model is None or self._torch is None:
            self.load()
        from PIL import Image

        image = Image.fromarray(np.asarray(step.image, dtype=np.uint8))
        prompt = _openvla_prompt(step.task_description, official=self.use_openvla_prompt)
        inputs = self._processor(prompt, image).to(self.device)
        with self._torch.no_grad():
            if not hasattr(self._model, "predict_action"):
                raise AdapterError(
                    "Loaded OpenVLA model does not expose predict_action(). "
                    "Use --adapter-factory for this checkpoint."
                )
            action = self._model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )
        return _to_action_array(action)


def _openvla_prompt(task_description: str, *, official: bool = False) -> str:
    task = task_description.strip()
    if official:
        return f"In: What action should the robot take to {task}?\nOut:"
    return f"What action should the robot take to {task}?"


def _build_vlm4vla_policy(request: PolicyBuildRequest) -> Any:
    root = Path(request.vlm4vla_root or "").expanduser()
    if not root.is_dir():
        raise AdapterError(f"--vlm4vla-root does not exist or is not a directory: {root}")
    _prepend_sys_path(root)
    if request.openpi_root:
        openpi_root = Path(request.openpi_root).expanduser()
        openpi_src = openpi_root / "src" if (openpi_root / "src").is_dir() else openpi_root
        _prepend_sys_path(openpi_src)
        os.environ.setdefault("OPENPI_SRC", str(openpi_src))

    try:
        import torch
        from eval.libero.model_wrapper import BaseModelInference
    except ImportError as exc:
        raise AdapterError(
            "Could not import VLM4VLA eval.libero.model_wrapper from --vlm4vla-root. "
            "Check that the path points to a VLM4VLA checkout and its environment is active."
        ) from exc

    configs = _load_vlm4vla_config(request)
    _inject_public_runtime_config(configs, request)
    ckpt_path = request.model_path or configs.get("ckpt_path")
    device = torch.device(request.device if torch.cuda.is_available() or request.device == "cpu" else "cpu")
    return BaseModelInference(
        ckpt_path=ckpt_path,
        configs=configs,
        device=device,
        save_dir=request.output_dir,
        unnorm_key=request.unnorm_key,
        execute_step=request.execute_step,
        policy_setup=request.dataset,
        center_crop=request.center_crop,
    )


def _load_vlm4vla_config(request: PolicyBuildRequest) -> dict[str, Any]:
    if request.config_path:
        try:
            from vlm4vla.utils.config_utils import load_config as vlm4vla_load_config

            loaded = vlm4vla_load_config(request.config_path)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return load_config(request.config_path)
    if request.model_config:
        return load_config(request.model_config)
    return {}


def _inject_public_runtime_config(configs: dict[str, Any], request: PolicyBuildRequest) -> None:
    configs.update(request.extra_config)
    configs.setdefault("model", "pi0" if request.model == "pi05" else "openvla")
    configs["model_path"] = request.model_path
    if request.model_path:
        configs.setdefault("vlm", {})
        if isinstance(configs["vlm"], dict):
            configs["vlm"]["pretrained_model_name_or_path"] = request.model_path
        if request.model == "pi05":
            configs["pi0_checkpoint_path"] = request.model_path
    if request.openpi_config_name:
        configs["openpi_config_name"] = request.openpi_config_name
    if request.tokenizer_path:
        configs["tokenizer_path"] = request.tokenizer_path
    if request.replan_steps is not None:
        configs["replan_steps"] = request.replan_steps
    if request.use_openvla_prompt:
        configs["use_openvla_prompt"] = True
    if request.single_unnorm:
        configs["norm_action"] = False
    if request.model == "pi05":
        configs["norm_action"] = False
        configs.setdefault("pi05", True)
    if request.knockout_config:
        configs["knockout_config"] = request.knockout_config


def _prepend_sys_path(path: str | Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
