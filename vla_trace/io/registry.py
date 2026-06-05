"""Public config registry for command-line shortcuts."""

from __future__ import annotations

MODEL_CONFIGS = {
    "openvla": "configs/models/openvla.yaml",
    "pi05": "configs/models/pi05.yaml",
}

MODEL_ALIASES = {
    "openvla": "openvla",
    "open-vla": "openvla",
    "OpenVLA": "openvla",
    "pi05": "pi05",
    "pi0.5": "pi05",
    "pi0_5": "pi05",
    "pi-0.5": "pi05",
}

DATASET_CONFIGS = {
    "libero_10": "configs/benchmarks/libero_10.yaml",
    "libero_goal": "configs/benchmarks/libero_goal.yaml",
    "libero_object": "configs/benchmarks/libero_object.yaml",
    "libero_spatial": "configs/benchmarks/libero_spatial.yaml",
}

DATASET_ALIASES = {
    "libero_10": "libero_10",
    "libero-10": "libero_10",
    "libero10": "libero_10",
    "libero_goal": "libero_goal",
    "libero-goal": "libero_goal",
    "goal": "libero_goal",
    "libero_object": "libero_object",
    "libero-object": "libero_object",
    "object": "libero_object",
    "libero_spatial": "libero_spatial",
    "libero-spatial": "libero_spatial",
    "spatial": "libero_spatial",
}

DEFAULT_TOKEN_LAYOUTS = {
    "openvla": {
        "visual_tokens": 256,
        "text_tokens": 32,
        "action_tokens": 7,
    },
    "pi05": {
        "visual_tokens": 768,
        "text_tokens": 32,
        "action_tokens": 10,
        "token_order": "text,visual,action",
    },
}

DEFAULT_NUM_LAYERS = {
    "openvla": 32,
    "pi05": 18,
}

DEFAULT_CENTER_LAYERS = {
    "openvla": [0, 8, 16, 24, 31],
    "pi05": [0, 4, 8, 12, 17],
}

DEFAULT_WINDOW_SIZE = {
    "openvla": 7,
    "pi05": 5,
}

DEFAULT_KNOCKOUT_MODE = {
    "openvla": "no_image",
    "pi05": "no_text",
}


def normalize_model(value: str) -> str:
    key = MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower()))
    if key is None:
        supported = ", ".join(["OpenVLA", "pi0.5"])
        raise ValueError(f"Unsupported model {value!r}. Choose one of: {supported}")
    return key


def normalize_dataset(value: str) -> str:
    key = DATASET_ALIASES.get(value, DATASET_ALIASES.get(value.lower()))
    if key is None:
        supported = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unsupported dataset {value!r}. Choose one of: {supported}")
    return key


def config_path(relative_path: str) -> str:
    return relative_path


def model_config_path(model: str) -> str:
    return config_path(MODEL_CONFIGS[normalize_model(model)])


def dataset_config_path(dataset: str) -> str:
    return config_path(DATASET_CONFIGS[normalize_dataset(dataset)])
