"""Example LIBERO policy adapter for OpenVLA-OFT-style checkpoints.

Use this when an OpenVLA-OFT checkout is on ``PYTHONPATH`` and you want
``vla-trace eval-libero`` to call that policy during baseline, PatchMask, or
custom knockout rollouts. Copy the file into your project if your checkpoint
uses different observation keys, action heads, or attention hooks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


def create_policy(request: Any) -> "OpenVLAOFTPolicy":
    return OpenVLAOFTPolicy(request)


class OpenVLAOFTPolicy:
    """Small adapter around the OpenVLA-OFT evaluation utilities."""

    def __init__(self, request: Any) -> None:
        if not request.model_path:
            raise ValueError("OpenVLA-OFT adapter requires --model-path")
        self.request = request
        self.knockout_config: dict[str, Any] | None = request.knockout_config
        self.cfg = self._build_cfg(request)
        self.vla = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None
        self.noisy_action_projector = None
        self._get_vla_action = None
        self._resize_image_for_policy = None
        self._action_queue: list[np.ndarray] = []
        self.load()

    def _build_cfg(self, request: Any) -> SimpleNamespace:
        extra = dict(getattr(request, "extra_config", {}) or {})
        return SimpleNamespace(
            pretrained_checkpoint=request.model_path,
            use_l1_regression=bool(extra.get("use_l1_regression", True)),
            use_diffusion=bool(extra.get("use_diffusion", False)),
            use_film=bool(extra.get("use_film", False)),
            num_images_in_input=int(extra.get("num_images_in_input", 2)),
            use_proprio=bool(extra.get("use_proprio", True)),
            load_in_8bit=bool(extra.get("load_in_8bit", False)),
            load_in_4bit=bool(extra.get("load_in_4bit", False)),
            center_crop=bool(getattr(request, "center_crop", False) or extra.get("center_crop", True)),
            num_open_loop_steps=int(extra.get("num_open_loop_steps", 8)),
            unnorm_key=getattr(request, "unnorm_key", None) or f"{request.dataset}_no_noops",
            lora_rank=int(extra.get("lora_rank", 32)),
            num_diffusion_steps_train=int(extra.get("num_diffusion_steps_train", 50)),
            num_diffusion_steps_inference=int(extra.get("num_diffusion_steps_inference", 50)),
        )

    def load(self) -> None:
        try:
            from experiments.robot import openvla_utils
            from prismatic.vla.constants import PROPRIO_DIM
        except ImportError as exc:
            raise RuntimeError(
                "OpenVLA-OFT adapter requires the OpenVLA-OFT repository and "
                "dependencies on PYTHONPATH."
            ) from exc

        self.vla = openvla_utils.get_vla(self.cfg)
        self.processor = openvla_utils.get_processor(self.cfg)
        self._get_vla_action = openvla_utils.get_vla_action
        self._resize_image_for_policy = getattr(openvla_utils, "resize_image_for_policy", None)
        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = openvla_utils.get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        if self.cfg.use_proprio:
            self.proprio_projector = openvla_utils.get_proprio_projector(
                self.cfg,
                llm_dim=self.vla.llm_dim,
                proprio_dim=PROPRIO_DIM,
            )
        if self.cfg.use_diffusion:
            get_noisy_action_projector = getattr(openvla_utils, "get_noisy_action_projector", None)
            if get_noisy_action_projector is None:
                raise RuntimeError("This OpenVLA-OFT checkout does not expose get_noisy_action_projector()")
            self.noisy_action_projector = get_noisy_action_projector(self.cfg, llm_dim=self.vla.llm_dim)

    def reset(self) -> None:
        self._action_queue.clear()

    def configure_knockout(self, config: dict[str, Any] | None) -> None:
        self.knockout_config = config
        if not config:
            return
        for target in (self.vla, getattr(self.vla, "model", None)):
            if hasattr(target, "configure_knockout"):
                target.configure_knockout(config)
                return
            if hasattr(target, "set_knockout_config"):
                target.set_knockout_config(config)
                return
        raise NotImplementedError(
            "Map VLA-Trace knockout_config to your OpenVLA-OFT attention hooks "
            "before running Stage 2 interventions with this adapter."
        )

    def predict_action(self, step: Any) -> np.ndarray:
        if self._action_queue:
            return self._action_queue.pop(0)
        if self._get_vla_action is None:
            self.load()

        observation = self._build_oft_observation(step)
        actions = self._get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            step.task_description,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
        )
        self._action_queue = [np.asarray(action, dtype=np.float32) for action in actions]
        if not self._action_queue:
            raise RuntimeError("OpenVLA-OFT policy returned an empty action chunk")
        return self._action_queue.pop(0)

    def _build_oft_observation(self, step: Any) -> dict[str, Any]:
        raw = dict(step.raw_observation)
        full_image = self._raw_libero_image(raw, "agentview_image")
        if full_image is None:
            full_image = np.asarray(step.image, dtype=np.uint8)
        observation = {
            "full_image": self._resize_openvla_image(full_image),
            "task_description": step.task_description,
        }
        if self.cfg.num_images_in_input > 1:
            if step.wrist_image is not None:
                wrist_image = np.asarray(step.wrist_image, dtype=np.uint8)
            else:
                wrist_image = self._raw_libero_image(raw, "robot0_eye_in_hand_image")
            if wrist_image is not None:
                observation["wrist_image"] = self._resize_openvla_image(wrist_image)
        if self.cfg.use_proprio:
            if step.state is not None:
                observation["state"] = np.asarray(step.state, dtype=np.float32)
            else:
                from vla_trace.evaluation.libero import get_libero_state

                observation["state"] = get_libero_state(raw)
        return observation

    def _raw_libero_image(self, raw: dict[str, Any], key: str) -> np.ndarray | None:
        if key not in raw:
            return None
        return np.asarray(raw[key], dtype=np.uint8)[::-1, ::-1]

    def _resize_openvla_image(self, image: np.ndarray) -> np.ndarray:
        if self._resize_image_for_policy is not None:
            return np.asarray(self._resize_image_for_policy(image, 224), dtype=np.uint8)
        return np.asarray(image, dtype=np.uint8)
