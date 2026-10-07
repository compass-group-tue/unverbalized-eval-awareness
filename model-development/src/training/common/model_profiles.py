"""Architecture-specific defaults for the training entry points.

Profiles deliberately contain no checkpoint paths. A caller must provide
``--model-path`` so the shared release neither assumes nor exposes a local
model store. Profiles select only architecture-sensitive LoRA and rendering
defaults.
"""

from __future__ import annotations

from typing import Any


DEFAULT_QWEN3_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3_32b": {
        "model_family": "qwen3",
        "attn_implementation": "sdpa",
        "target_modules": DEFAULT_QWEN3_TARGET_MODULES,
        "sft_thinking_control_placement": "system",
        "sft_reasoning_control": "/think",
        "sft_non_reasoning_control": "/no_think",
        "sft_pass_enable_thinking": True,
    },
    "qwen3_5_27b": {
        "model_family": "qwen3_5",
        "attn_implementation": "sdpa",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        "sft_thinking_control_placement": "system",
        "sft_reasoning_control": "",
        "sft_non_reasoning_control": "",
        "sft_pass_enable_thinking": True,
    },
    "gemma4_31b": {
        "model_family": "gemma4",
        "attn_implementation": None,
        "target_modules": "all-linear",
        "sft_thinking_control_placement": "off",
        "sft_reasoning_control": "",
        "sft_non_reasoning_control": "",
        "sft_pass_enable_thinking": False,
    },
}
