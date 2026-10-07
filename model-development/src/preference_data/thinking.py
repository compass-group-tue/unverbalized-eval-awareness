"""Model-family thinking controls shared by Stage-1 generation runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThinkingControlConfig:
    model_family: str
    supports_thinking_soft_switch: bool
    supports_enable_thinking: bool
    effective_reasoning_control: str
    vllm_reasoning_parser: str | None


def infer_thinking_control_config(
    *, base_model_path: str, requested_reasoning_control: str
) -> ThinkingControlConfig:
    """Infer the compatible thinking controls from an explicit model path."""
    model_name = (base_model_path or "").strip().lower()
    requested_control = (requested_reasoning_control or "").strip()

    if any(token in model_name for token in ("qwen3.5", "qwen3_5", "qwen3-5")):
        return ThinkingControlConfig(
            model_family="qwen3_5",
            supports_thinking_soft_switch=False,
            supports_enable_thinking=True,
            effective_reasoning_control="",
            vllm_reasoning_parser="qwen3",
        )
    if "gemma-4" in model_name or "gemma4" in model_name:
        return ThinkingControlConfig(
            model_family="gemma4",
            supports_thinking_soft_switch=False,
            supports_enable_thinking=False,
            effective_reasoning_control="",
            vllm_reasoning_parser=None,
        )
    return ThinkingControlConfig(
        model_family="default",
        supports_thinking_soft_switch=True,
        supports_enable_thinking=True,
        effective_reasoning_control=requested_control,
        vllm_reasoning_parser=None,
    )
