#!/usr/bin/env python3
"""Assistant-span DPO for the special-token cleanup preference dataset.

The prepared rows contain complete chosen/rejected chat trajectories.  This
trainer deliberately scores every assistant span rather than treating the
trajectory as one contiguous completion after a prompt, which is required for
the MT Bench rows.  The objective is conventional sigmoid DPO with frozen
reference sequence-logprob sums precomputed in separate shard jobs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

from training.common.model_profiles import MODEL_PROFILES
from training.dpo.kto import kto_loss_from_logratios
from training.dpo.td_alignment import (
    annotate_segment,
    apply_visible_difference_multiplier,
    normalize_branch_weights,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


LABEL_PAD_TOKEN_ID = -100
ASSISTANT_LOSS_MASK_CHOICES = ("all_assistant", "content_only")
# ``legacy_prefix_delta`` reproduces every pre-V13 run: the native Qwen
# inference template is rendered repeatedly and assistant labels are inferred
# from token-prefix deltas.  ``trl_qwen3_5_training`` uses TRL's pinned,
# Qwen3.5-specific training template and its direct assistant-token mask.
# The latter still selects only the final _dpo_trainable completion in each
# prepared segment; earlier assistant turns are context, not targets.
CHAT_RENDERING_CHOICES = ("legacy_prefix_delta", "trl_qwen3_5_training")
LEGACY_CHAT_RENDERING = "legacy_prefix_delta"
# CLI default for SFT/DPO training and reference scoring: the paper model (V13).
# Library functions keep the legacy default, and reference caches written
# before the field existed are read as legacy.
DEFAULT_CHAT_RENDERING = "trl_qwen3_5_training"
TRL_TRAINING_TEMPLATE_VERSION = "1.9.0"
CHOSEN_NLL_SCALE_MODES = ("fixed", "dpo_saturation")
TD_DPO_MODES = ("none", "symmetric")
PREFERENCE_OBJECTIVES = ("dpo", "kto")
CHECKPOINT_COMPLETE_MARKER = ".checkpoint_complete"


def distributed_barrier() -> None:
    """Synchronize ranks only when this process is in an initialized group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


class DetailedTimingRecorder:
    """Small per-rank JSONL recorder for diagnosing end-to-end train cadence."""

    def __init__(self, output_dir: str | None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.rank = int(os.environ.get("RANK", "0"))
        self._inner_batches: list[dict[str, Any]] = []
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def _write(self, record: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        record = {
            "rank": self.rank,
            "wall_time_unix": time.time(),
            "monotonic_seconds": time.perf_counter(),
            **record,
        }
        path = self.output_dir / f"rank_{self.rank:03d}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def event(self, name: str, **fields: Any) -> None:
        self._write({"record_type": "lifecycle", "event": name, **fields})

    def record_inner_batch(self, *, pair_ids: list[str], elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        self._inner_batches.append(
            {
                "pair_ids": pair_ids,
                "elapsed_seconds_forward_backward": elapsed_seconds,
            }
        )

    def consume_inner_batches(self) -> list[dict[str, Any]]:
        result = self._inner_batches
        self._inner_batches = []
        return result


class DetailedStepTimingCallback(TrainerCallback):
    """Record wall-clock time between completed optimizer steps."""

    def __init__(self, recorder: DetailedTimingRecorder) -> None:
        self.recorder = recorder
        self.loop_started: float | None = None
        self.previous_step_ended: float | None = None
        self.current_step_started: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        self.loop_started = time.perf_counter()
        self.recorder.event("trainer_loop_begin", global_step=state.global_step)
        return control

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        self.current_step_started = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        ended = time.perf_counter()
        self.recorder._write(
            {
                "record_type": "optimizer_step",
                "global_step": state.global_step,
                "seconds_since_trainer_loop_begin": (
                    ended - self.loop_started if self.loop_started is not None else None
                ),
                "seconds_since_previous_step_end": (
                    ended - self.previous_step_ended if self.previous_step_ended is not None else None
                ),
                "seconds_from_step_begin_to_step_end": (
                    ended - self.current_step_started if self.current_step_started is not None else None
                ),
                "inner_forward_backward_batches": self.recorder.consume_inner_batches(),
            }
        )
        self.previous_step_ended = ended
        self.current_step_started = None
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        self.recorder.event("trainer_loop_end", global_step=state.global_step)
        return control


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("precompute-reference", "attach-reference", "train"), required=True)
    p.add_argument("--data-jsonl", required=True, help="Prepared DPO train or validation JSONL.")
    p.add_argument(
        "--output-path",
        default=None,
        help="Required for precompute-reference and attach-reference modes.",
    )
    p.add_argument("--reference-jsonl", action="append", default=[])
    p.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="qwen3_5_27b")
    p.add_argument("--model-path", default=None)
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--target-modules", nargs="+", default=None)
    p.add_argument("--max-length", type=int, default=10240)
    p.add_argument(
        "--assistant-loss-mask",
        choices=ASSISTANT_LOSS_MASK_CHOICES,
        default="content_only",
        help=(
            "Tokens contributing to assistant-span log-probabilities. content_only (default; the paper "
            "model) labels only rendered reasoning_content/content text, excluding template formatting; "
            "all_assistant also includes native chat-template headers and boundaries."
        ),
    )
    p.add_argument(
        "--chat-rendering",
        choices=CHAT_RENDERING_CHOICES,
        default=DEFAULT_CHAT_RENDERING,
        help=(
            "trl_qwen3_5_training (default; the paper model) uses TRL "
            "1.9.0's Qwen3.5 prefix-preserving training template and direct assistant mask; "
            "it requires one final _dpo_trainable assistant per prepared segment and fresh "
            "reference log-probabilities. legacy_prefix_delta reproduces pre-V13 runs."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--precompute-batch-size", type=int, default=1)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument(
        "--preference-objective",
        choices=PREFERENCE_OBJECTIVES,
        default="dpo",
        help=(
            "Preference objective. dpo preserves the existing paired sigmoid-DPO path and its precomputed "
            "reference scores. kto converts the two branches into balanced desirable/undesirable examples and "
            "uses an online, cyclically mismatched KTO reference-point estimate."
        ),
    )
    p.add_argument(
        "--kto-desirable-weight",
        type=float,
        default=1.0,
        help="KTO lambda_D multiplier for desirable (chosen) examples.",
    )
    p.add_argument(
        "--kto-undesirable-weight",
        type=float,
        default=1.0,
        help="KTO lambda_U multiplier for undesirable (rejected) examples.",
    )
    p.add_argument(
        "--kto-kl-score-batch-size",
        type=int,
        default=0,
        help=(
            "Maximum number of detached cyclic KTO pseudo-sequences to score in one forward. "
            "0 preserves the unchunked batch. Smaller positive values reduce the fp32 log-softmax peak "
            "without changing which pseudo-sequences contribute to the KTO reference-point estimate."
        ),
    )
    p.add_argument(
        "--chosen-nll-coef",
        type=float,
        default=0.05,
        help=(
            "Coefficient for a length-normalized NLL term on the chosen trajectory (default 0.05, the "
            "paper model; 0 disables it, as KTO and --dpop-penalty-weight require). "
            "Its token mask is selected independently by --chosen-nll-mask, while "
            "--assistant-loss-mask continues to define the DPO/reference objective."
        ),
    )
    p.add_argument(
        "--chosen-nll-mask",
        choices=ASSISTANT_LOSS_MASK_CHOICES,
        default="all_assistant",
        help=(
            "Assistant-token mask for the optional chosen NLL regularizer. Defaults to all_assistant "
            "to preserve the historical NLL behaviour; content_only makes it match the DPO mask."
        ),
    )
    p.add_argument(
        "--chosen-nll-scale-mode",
        choices=CHOSEN_NLL_SCALE_MODES,
        default="fixed",
        help=(
            "How the chosen-NLL coefficient is scaled over training. fixed preserves the historical "
            "constant coefficient. dpo_saturation multiplies it by an EMA of sigmoid(-dpo_logit), so the "
            "regularizer decays in lockstep with the DPO term instead of dominating once DPO saturates."
        ),
    )
    p.add_argument(
        "--chosen-nll-scale-ema",
        type=float,
        default=0.99,
        help="EMA momentum for --chosen-nll-scale-mode dpo_saturation. Higher is smoother.",
    )
    p.add_argument(
        "--length-normalize-dpo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Divide each side's summed assistant log-probability (policy and reference alike) by its "
            "scored-token count before the DPO comparison. This keeps the preference logit in per-token "
            "units so it does not saturate purely because completions are long. Requires a larger --beta."
        ),
    )
    p.add_argument(
        "--td-dpo-mode",
        choices=TD_DPO_MODES,
        default="none",
        help=(
            "Optional difference-aware DPO weighting. symmetric upweights exact content-token differences on "
            "both chosen and rejected trajectories and downweights exact shared tokens. none preserves "
            "ordinary DPO exactly."
        ),
    )
    p.add_argument(
        "--td-dpo-difference-weight",
        type=float,
        default=2.0,
        help="Unnormalized TD-DPO weight for a content token not exactly shared by both branches.",
    )
    p.add_argument(
        "--td-dpo-shared-weight",
        type=float,
        default=0.5,
        help="Unnormalized TD-DPO weight for an exactly shared content token.",
    )
    p.add_argument(
        "--td-dpo-normalize-branch-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rescale each complete chosen/rejected trajectory's TD weights to mean one over selected content "
            "tokens. This preserves the ordinary DPO branch scale while retaining relative difference/shared "
            "emphasis. Has no effect when --td-dpo-mode=none."
        ),
    )
    p.add_argument(
        "--td-dpo-visible-difference-multiplier",
        type=float,
        default=1.0,
        help=(
            "Additional relative multiplier for TD-DPO tokens that are both exact branch differences "
            "and part of the final visible assistant response. Applied before optional branch "
            "normalization; 1.0 preserves ordinary symmetric TD-DPO."
        ),
    )
    p.add_argument(
        "--dpop-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Optional DPO-Positive penalty weight. A positive value penalizes decreases in chosen "
            "log-probability relative to the frozen reference. Mutually exclusive with --chosen-nll-coef."
        ),
    )
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--save-total-limit", type=int, default=5)
    p.add_argument(
        "--ddp-timeout-seconds",
        type=int,
        default=3600,
        help=(
            "Distributed-process timeout passed to TrainingArguments. It must exceed the longest expected "
            "rank-zero full-checkpoint write when checkpoint synchronization is enabled."
        ),
    )
    p.add_argument(
        "--synchronize-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Barrier all ranks after full checkpoints and adapter-only snapshots so nonzero ranks cannot "
            "start the next DDP collective while rank zero is still writing to storage."
        ),
    )
    p.add_argument("--eval-jsonl", default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--output-dir", default=None, help="Checkpoint directory; required for train mode.")
    p.add_argument("--final-adapter-dir", default=None, help="Required for train mode.")
    p.add_argument(
        "--adapter-snapshot-steps",
        default="",
        help=(
            "Optional comma-separated global steps at which to write an adapter-only snapshot, in "
            "addition to the regular --save-steps checkpoints. Intended for dense early candidates when "
            "the objective is expected to saturate long before the end of the epoch. These carry no "
            "optimizer state and cannot be resumed from; merge and evaluate them instead."
        ),
    )
    p.add_argument(
        "--adapter-snapshot-dir",
        default=None,
        help="Destination for --adapter-snapshot-steps. Defaults to <output-dir>/../adapter_snapshots.",
    )
    p.add_argument(
        "--consumed-pairs-dir",
        default=None,
        help=(
            "Optional directory recording which pair_ids each rank consumed at each optimizer step. "
            "Use it to reconstruct the exact training prefix behind any checkpoint, and to derive the "
            "unconsumed remainder as a held-out pool for later expert iteration."
        ),
    )
    p.add_argument(
        "--step-timing-dir",
        default=None,
        help="Optional directory for one JSONL timing record per local DPO batch.",
    )
    p.add_argument(
        "--detailed-timing-dir",
        default=None,
        help=(
            "Optional smoke/debug directory for lifecycle and complete optimizer-step timing. "
            "Disabled by default for production runs."
        ),
    )
    p.add_argument(
        "--report-to",
        choices=("none", "wandb"),
        default="none",
        help="Optional Trainer reporting backend. W&B configuration comes from environment variables.",
    )
    p.add_argument("--run-name", default=None, help="Optional human-readable reporting run name.")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--reference-check-examples", type=int, default=8)
    p.add_argument("--reference-tolerance", type=float, default=0.1)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield item


def row_has_empty_trainable_assistant(row: dict[str, Any]) -> bool:
    """Whether a DPO pair has a completion with no real assistant text at all."""
    for side in ("chosen_segments", "rejected_segments"):
        segments = row.get(side)
        if not isinstance(segments, list):
            continue
        for messages in segments:
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") != "assistant" or message.get("_dpo_trainable") is not True:
                    continue
                if str(message.get("reasoning_content") or "") or str(message.get("content") or ""):
                    continue
                return True
    return False


def load_rows(
    path: Path,
    max_examples: int = 0,
    *,
    assistant_loss_mask: str = "all_assistant",
) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if assistant_loss_mask == "content_only":
        original_count = len(rows)
        rows = [row for row in rows if not row_has_empty_trainable_assistant(row)]
        removed = original_count - len(rows)
        if removed:
            print(
                f"[data] content_only removed {removed} pair(s) with an empty trainable assistant completion "
                f"from {path}",
                flush=True,
            )
    if max_examples > 0:
        rows = rows[:max_examples]
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_profile(args: argparse.Namespace) -> argparse.Namespace:
    profile = MODEL_PROFILES[args.model_profile]
    if args.model_path is None:
        raise ValueError("--model-path is required; model profiles contain architecture defaults only.")
    if args.attn_implementation is None:
        args.attn_implementation = profile["attn_implementation"]
    if args.target_modules is None:
        modules = profile["target_modules"]
        args.target_modules = [modules] if isinstance(modules, str) else list(modules)
    return args


def get_1d_ids(rendered: Any) -> list[int]:
    # Transformers returns a BatchEncoding here, which is mapping-like but is
    # not necessarily an instance of the concrete ``dict`` class.
    ids = rendered.get("input_ids") if hasattr(rendered, "get") else None
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        raise ValueError("chat template did not return input_ids")
    return [int(token) for token in ids]


def get_1d_assistant_mask(rendered: Any) -> list[bool]:
    mask = rendered.get("assistant_masks") if hasattr(rendered, "get") else None
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if isinstance(mask, list) and mask and isinstance(mask[0], list):
        mask = mask[0]
    if not isinstance(mask, list):
        raise ValueError("chat template did not return assistant_masks")
    return [bool(value) for value in mask]


def get_1d_offsets(rendered: Any) -> list[tuple[int, int]]:
    offsets = rendered.get("offset_mapping") if hasattr(rendered, "get") else None
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if isinstance(offsets, list) and offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    if not isinstance(offsets, list):
        raise ValueError("tokenizer did not return offset mappings")
    result: list[tuple[int, int]] = []
    for offset in offsets:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("tokenizer returned malformed offset mappings")
        result.append((int(offset[0]), int(offset[1])))
    return result


def apply_native_chat_template(tokenizer, messages: list[dict[str, Any]], *, tokenize: bool) -> Any:
    """Render through the native template with identical controls in both paths."""
    kwargs: dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": False,
        "enable_thinking": True,
    }
    if tokenize:
        kwargs["return_dict"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # If this optional control is unavailable, both the text and token
        # paths take this exact fallback, preserving internal consistency.
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def get_trl_qwen3_5_training_template(tokenizer) -> str:
    """Return TRL's pinned Qwen3.5 training template, once per tokenizer.

    The template is deliberately supplied by TRL rather than copied into this
    repo.  It adds generation markers for direct assistant-loss masks and fixes
    Qwen3.5's non-prefix-preserving treatment of earlier thinking blocks.
    Version pinning prevents an environment upgrade from silently changing the
    DPO/reference scoring contract.
    """
    cached = getattr(tokenizer, "_special_token_trl_qwen3_5_training_template", None)
    if isinstance(cached, str):
        return cached
    try:
        actual_version = package_version("trl")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "chat_rendering=trl_qwen3_5_training requires trl==1.9.0 in the active environment. "
            "Install it with uv pip install --no-deps 'trl==1.9.0'."
        ) from exc
    if actual_version != TRL_TRAINING_TEMPLATE_VERSION:
        raise RuntimeError(
            "chat_rendering=trl_qwen3_5_training is pinned to "
            f"trl=={TRL_TRAINING_TEMPLATE_VERSION}, but the active environment has trl=={actual_version}. "
            "Use the pinned version so the template and reference cache remain reproducible."
        )
    try:
        from trl.chat_template_utils import get_training_chat_template
    except ImportError as exc:
        raise RuntimeError(
            "trl is installed but its training-template utility could not be imported. "
            "Reinstall the pinned trl==1.9.0 wheel."
        ) from exc
    template = get_training_chat_template(tokenizer)
    if not isinstance(template, str) or not template:
        raise RuntimeError(
            "TRL did not provide a Qwen3.5 training template for this tokenizer. "
            "This rendering mode is intentionally restricted to supported Qwen3.5 tokenizers."
        )
    setattr(tokenizer, "_special_token_trl_qwen3_5_training_template", template)
    return template


def apply_trl_qwen3_5_training_template(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
    return_assistant_tokens_mask: bool = False,
) -> Any:
    """Render through TRL's official Qwen3.5 training template."""
    kwargs: dict[str, Any] = {
        "chat_template": get_trl_qwen3_5_training_template(tokenizer),
        "tokenize": tokenize,
        "add_generation_prompt": False,
        "enable_thinking": True,
    }
    if tokenize:
        kwargs["return_dict"] = True
        kwargs["return_assistant_tokens_mask"] = return_assistant_tokens_mask
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def _last_assistant_content_ranges(rendered_text: str, message: dict[str, Any]) -> list[tuple[int, int]]:
    """Find the final assistant message's real text in its rendered prefix.

    The Qwen template can revise earlier *formatting* when a later turn is
    appended, so rendered prefixes are not necessarily literal string prefixes
    of the final transcript. The existing trainer intentionally uses token
    prefix deltas for this reason. Here we locate the final assistant's text in
    its own prefix (where it is the final message), then reuse those same token
    positions in the established prefix-delta coordinate system.
    """
    reasoning = str(message.get("reasoning_content") or "")
    response = str(message.get("content") or "")
    if not reasoning and not response:
        raise ValueError("content-only mask found an empty trainable assistant message")

    response_start = rendered_text.rfind(response) if response else len(rendered_text)
    if response and response_start < 0:
        raise ValueError("could not locate trainable assistant content in its native rendered prefix")
    reasoning_start = rendered_text.rfind(reasoning, 0, response_start) if reasoning else -1
    if reasoning and reasoning_start < 0:
        raise ValueError("could not locate trainable assistant reasoning_content in its native rendered prefix")

    ranges: list[tuple[int, int]] = []
    if reasoning:
        ranges.append((reasoning_start, reasoning_start + len(reasoning)))
    if response:
        ranges.append((response_start, response_start + len(response)))
    return ranges


def _content_only_masks_for_prefix(
    tokenizer,
    partial_ids: list[int],
    partial_text: str,
    message: dict[str, Any],
) -> tuple[list[bool], list[bool]]:
    """Return masks for final assistant text and its visible-response subset."""
    encoded = tokenizer(partial_text, add_special_tokens=False, return_offsets_mapping=True)
    text_ids = get_1d_ids(encoded)
    offsets = get_1d_offsets(encoded)
    if text_ids != partial_ids:
        raise ValueError(
            "native chat-template IDs differ from tokenizing its rendered prefix; "
            "cannot safely build a content-only assistant mask"
        )
    if len(offsets) != len(partial_ids):
        raise ValueError("offset mapping length does not match native chat-template IDs")
    content_ranges = _last_assistant_content_ranges(partial_text, message)
    response = str(message.get("content") or "")
    response_start = partial_text.rfind(response) if response else -1
    response_range = (response_start, response_start + len(response)) if response else None
    mask = [False] * len(partial_ids)
    visible_mask = [False] * len(partial_ids)
    for position, (start, end) in enumerate(offsets):
        # Zero-width offsets are special/template tokens. A token overlapping
        # real content remains trainable even if it includes surrounding space.
        if any(start < range_end and end > range_start for range_start, range_end in content_ranges):
            mask[position] = True
        if response_range is not None and start < response_range[1] and end > response_range[0]:
            visible_mask[position] = True
    if not any(mask):
        raise ValueError("content-only mask rendered no trainable assistant tokens")
    if any(visible_mask[position] and not mask[position] for position in range(len(mask))):
        raise AssertionError("visible response mask escaped content-only assistant mask")
    return mask, visible_mask


def final_trainable_assistant_index(messages: list[dict[str, Any]]) -> int:
    """Validate the one-completion contract used by the TRL rendering mode."""
    trainable = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("_dpo_trainable") is True
    ]
    if len(trainable) != 1 or trainable[0] != len(messages) - 1:
        raise ValueError(
            "chat_rendering=trl_qwen3_5_training requires exactly one final _dpo_trainable assistant "
            f"message per prepared segment, found positions {trainable} among {len(messages)} messages"
        )
    return trainable[0]


def render_sequence_trl_qwen3_5_training(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    assistant_loss_mask: str,
    return_visible_content_mask: bool = False,
) -> tuple[list[int], list[int]] | tuple[list[int], list[int], list[bool]]:
    """Render one final completion with TRL's direct Qwen3.5 assistant mask.

    The training template is prefix-preserving.  We therefore use one prompt
    render only to establish the final-completion boundary, then take the
    direct assistant mask on the suffix.  This prevents labels on assistant
    history while avoiding the legacy repeated-prefix-delta attribution.
    """
    final_index = final_trainable_assistant_index(messages)
    template_messages = [{key: value for key, value in message.items() if key != "_dpo_trainable"} for message in messages]
    full = apply_trl_qwen3_5_training_template(
        tokenizer, template_messages, tokenize=True, return_assistant_tokens_mask=True
    )
    full_ids = get_1d_ids(full)
    full_assistant_mask = get_1d_assistant_mask(full)
    if len(full_ids) != len(full_assistant_mask):
        raise ValueError("TRL training template returned input_ids and assistant_masks with different lengths")
    prompt_ids = get_1d_ids(
        apply_trl_qwen3_5_training_template(
            tokenizer, template_messages[:final_index], tokenize=True, return_assistant_tokens_mask=False
        )
    )
    if len(prompt_ids) >= len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        divergence = next(
            (
                position
                for position in range(min(len(prompt_ids), len(full_ids)))
                if prompt_ids[position] != full_ids[position]
            ),
            min(len(prompt_ids), len(full_ids)),
        )
        raise ValueError(
            "TRL Qwen3.5 training-template prompt is not a token prefix of the full segment "
            f"(prompt_tokens={len(prompt_ids)}, full_tokens={len(full_ids)}, first_divergence={divergence})"
        )

    selected_assistant = [
        position >= len(prompt_ids) and selected
        for position, selected in enumerate(full_assistant_mask)
    ]
    if any(selected_assistant[: len(prompt_ids)]):
        raise AssertionError("TRL final-completion mask selected an assistant-history token")

    labels = [LABEL_PAD_TOKEN_ID] * len(full_ids)
    visible_content_mask = [False] * len(full_ids)
    if assistant_loss_mask == "content_only" or return_visible_content_mask:
        rendered_text = apply_trl_qwen3_5_training_template(tokenizer, template_messages, tokenize=False)
        if not isinstance(rendered_text, str):
            raise ValueError("TRL training template did not return rendered text")
        content_mask, response_mask = _content_only_masks_for_prefix(
            tokenizer, full_ids, rendered_text, messages[final_index]
        )
        for position, token in enumerate(full_ids):
            if selected_assistant[position] and (
                assistant_loss_mask != "content_only" or content_mask[position]
            ):
                labels[position] = token
            if return_visible_content_mask and selected_assistant[position] and response_mask[position]:
                visible_content_mask[position] = True
    else:
        for position, token in enumerate(full_ids):
            if selected_assistant[position]:
                labels[position] = token

    if not any(label != LABEL_PAD_TOKEN_ID for label in labels):
        raise ValueError("TRL training rendering produced no trainable assistant tokens")
    if return_visible_content_mask:
        if any(
            visible and label == LABEL_PAD_TOKEN_ID
            for visible, label in zip(visible_content_mask, labels)
        ):
            raise AssertionError("TRL visible-response token escaped the selected assistant loss mask")
        return full_ids, labels, visible_content_mask
    return full_ids, labels


def render_sequence(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    assistant_loss_mask: str,
    return_visible_content_mask: bool = False,
    chat_rendering: str = LEGACY_CHAT_RENDERING,
) -> tuple[list[int], list[int]] | tuple[list[int], list[int], list[bool]]:
    if assistant_loss_mask not in ASSISTANT_LOSS_MASK_CHOICES:
        raise ValueError(f"Unknown assistant loss mask: {assistant_loss_mask}")
    if chat_rendering not in CHAT_RENDERING_CHOICES:
        raise ValueError(f"Unknown chat rendering mode: {chat_rendering}")
    if chat_rendering == "trl_qwen3_5_training":
        return render_sequence_trl_qwen3_5_training(
            tokenizer,
            messages,
            assistant_loss_mask=assistant_loss_mask,
            return_visible_content_mask=return_visible_content_mask,
        )
    template_messages = [{key: value for key, value in message.items() if key != "_dpo_trainable"} for message in messages]
    all_ids = get_1d_ids(apply_native_chat_template(tokenizer, template_messages, tokenize=True))
    # Original v1/v2 behavior: every token appended by a trainable assistant,
    # including native formatting delimiters, receives loss. content_only uses
    # the exact same prefix-delta coordinate system but retains only real
    # reasoning/response token positions in each trainable assistant prefix.
    labels = [LABEL_PAD_TOKEN_ID] * len(all_ids)
    visible_content_mask = [False] * len(all_ids)
    previous: list[int] = []
    seen_user = False
    for i, message in enumerate(messages):
        seen_user = seen_user or message.get("role") == "user"
        if not seen_user:
            continue
        partial_messages = template_messages[: i + 1]
        partial = get_1d_ids(apply_native_chat_template(tokenizer, partial_messages, tokenize=True))
        if message.get("role") == "assistant" and message.get("_dpo_trainable") is True:
            if assistant_loss_mask == "content_only" or return_visible_content_mask:
                partial_text = apply_native_chat_template(tokenizer, partial_messages, tokenize=False)
                if not isinstance(partial_text, str):
                    raise ValueError("partial chat template did not return rendered text")
                content_mask, response_mask = _content_only_masks_for_prefix(tokenizer, partial, partial_text, message)
                for position in range(len(previous), min(len(partial), len(all_ids))):
                    if assistant_loss_mask != "content_only" or content_mask[position]:
                        labels[position] = all_ids[position]
                    if return_visible_content_mask and response_mask[position]:
                        visible_content_mask[position] = True
            else:
                for position in range(len(previous), min(len(partial), len(all_ids))):
                    labels[position] = all_ids[position]
        previous = partial
    if not any(label != LABEL_PAD_TOKEN_ID for label in labels):
        raise ValueError("rendered sequence has no assistant tokens")
    if return_visible_content_mask:
        if any(visible_content_mask[position] and labels[position] == LABEL_PAD_TOKEN_ID for position in range(len(labels))):
            raise AssertionError("visible-response token is not selected by the DPO label mask")
        return all_ids, labels, visible_content_mask
    return all_ids, labels


def td_dpo_reference_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Objective fields that must agree between a reference cache and policy."""
    return {
        "td_dpo_mode": str(args.td_dpo_mode),
        "td_dpo_difference_weight": float(args.td_dpo_difference_weight),
        "td_dpo_shared_weight": float(args.td_dpo_shared_weight),
        "td_dpo_normalize_branch_weights": bool(args.td_dpo_normalize_branch_weights),
        "td_dpo_visible_difference_multiplier": float(args.td_dpo_visible_difference_multiplier),
    }


def chat_rendering_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Rendering choices that determine token-level scores and manifests."""
    return {
        "chat_rendering": str(args.chat_rendering),
        "trl_training_template_version": (
            TRL_TRAINING_TEMPLATE_VERSION if args.chat_rendering == "trl_qwen3_5_training" else None
        ),
    }


def validate_chat_rendering_reference_metadata(item: dict[str, Any], args: argparse.Namespace, *, source: str) -> None:
    """Reject caches scored with another template/mask coordinate system."""
    observed_rendering = str(item.get("chat_rendering", LEGACY_CHAT_RENDERING))
    if observed_rendering != args.chat_rendering:
        raise ValueError(
            f"{source}: reference used chat_rendering={observed_rendering!r}, expected {args.chat_rendering!r}. "
            "Reference log-probabilities must be recomputed whenever chat rendering changes."
        )
    expected_version = (
        TRL_TRAINING_TEMPLATE_VERSION if args.chat_rendering == "trl_qwen3_5_training" else None
    )
    observed_version = item.get("trl_training_template_version")
    if observed_version != expected_version:
        raise ValueError(
            f"{source}: reference used trl_training_template_version={observed_version!r}, "
            f"expected {expected_version!r}. Reference log-probabilities must be recomputed."
        )


def split_kto_prompt_and_completion(messages: list[dict[str, Any]], *, source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split one scored assistant turn into a prompt and completion for KTO.

    Each prepared trajectory segment is expected to end in exactly one trainable
    assistant turn.  We make that condition explicit rather than silently
    treating a multi-turn trajectory as one completion: the KTO KL estimate
    cyclically swaps precisely those assistant turns between prompt contexts.
    """
    trainable_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("_dpo_trainable") is True
    ]
    if len(trainable_positions) != 1:
        raise RuntimeError(
            f"{source}: KTO requires exactly one trainable assistant turn per prepared segment, "
            f"found {len(trainable_positions)}"
        )
    position = trainable_positions[0]
    if position != len(messages) - 1:
        raise RuntimeError(f"{source}: KTO trainable assistant turn must end its prepared segment")
    prompt = [{**message, "_dpo_trainable": False} for message in messages[:position]]
    completion = {**messages[position], "_dpo_trainable": True}
    return prompt, completion


def validate_td_dpo_reference_metadata(item: dict[str, Any], args: argparse.Namespace, *, source: str) -> None:
    """Reject a cache scored with a different TD-DPO statistic.

    Historical unweighted caches have no TD fields.  They remain valid only for
    the explicit ``none`` mode and are treated as its default metadata.
    """
    expected = td_dpo_reference_metadata(args)
    observed = {
        "td_dpo_mode": str(item.get("td_dpo_mode", "none")),
        "td_dpo_difference_weight": float(item.get("td_dpo_difference_weight", 2.0)),
        "td_dpo_shared_weight": float(item.get("td_dpo_shared_weight", 0.5)),
        "td_dpo_normalize_branch_weights": bool(item.get("td_dpo_normalize_branch_weights", True)),
        "td_dpo_visible_difference_multiplier": float(item.get("td_dpo_visible_difference_multiplier", 1.0)),
    }
    if observed != expected:
        raise ValueError(
            f"{source}: reference TD-DPO objective mismatch: observed={observed}, expected={expected}. "
            "Reference log-probabilities must be recomputed whenever TD-DPO scoring changes."
        )


class PreparedPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        assistant_loss_mask: str,
        *,
        chosen_nll_mask: str | None = None,
        td_dpo_mode: str = "none",
        td_dpo_difference_weight: float = 2.0,
        td_dpo_shared_weight: float = 0.5,
        td_dpo_normalize_branch_weights: bool = True,
        td_dpo_visible_difference_multiplier: float = 1.0,
        kto_enabled: bool = False,
        chat_rendering: str = LEGACY_CHAT_RENDERING,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_loss_mask = assistant_loss_mask
        self.chosen_nll_mask = chosen_nll_mask
        self.td_dpo_mode = td_dpo_mode
        self.td_dpo_difference_weight = td_dpo_difference_weight
        self.td_dpo_shared_weight = td_dpo_shared_weight
        self.td_dpo_normalize_branch_weights = td_dpo_normalize_branch_weights
        self.td_dpo_visible_difference_multiplier = td_dpo_visible_difference_multiplier
        self.kto_enabled = kto_enabled
        self.chat_rendering = chat_rendering

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        def tokenize_segments(key: str, loss_mask: str) -> list[dict[str, torch.Tensor]]:
            raw_segments = row.get(key)
            if not isinstance(raw_segments, list) or not raw_segments:
                raise RuntimeError(f"{row.get('pair_id')}: missing {key}")
            tokenized: list[dict[str, torch.Tensor]] = []
            for segment_index, messages in enumerate(raw_segments):
                if not isinstance(messages, list):
                    raise RuntimeError(f"{row.get('pair_id')}: {key}[{segment_index}] is not a message list")
                rendered = render_sequence(
                    self.tokenizer,
                    messages,
                    assistant_loss_mask=loss_mask,
                    return_visible_content_mask=self.td_dpo_mode == "symmetric",
                    chat_rendering=self.chat_rendering,
                )
                if self.td_dpo_mode == "symmetric":
                    input_ids, labels, visible_content_mask = rendered
                else:
                    input_ids, labels = rendered
                if len(input_ids) > self.max_length:
                    raise RuntimeError(
                        f"{row.get('pair_id')}: {key}[{segment_index}] exceeds --max-length {self.max_length}; "
                        "the prepared artifact and trainer configuration disagree."
                    )
                tokenized_segment: dict[str, Any] = {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
                if self.td_dpo_mode == "symmetric":
                    tokenized_segment["td_dpo_visible_content_mask"] = torch.tensor(
                        visible_content_mask, dtype=torch.bool
                    )
                if self.kto_enabled:
                    prompt, completion = split_kto_prompt_and_completion(
                        messages,
                        source=f"{row.get('pair_id')}:{key}[{segment_index}]",
                    )
                    tokenized_segment["kto_prompt_messages"] = prompt
                    tokenized_segment["kto_completion_message"] = completion
                tokenized.append(tokenized_segment)
            return tokenized
        out: dict[str, Any] = {
            "pair_id": str(row["pair_id"]),
            "chosen_segments": tokenize_segments("chosen_segments", self.assistant_loss_mask),
            "rejected_segments": tokenize_segments("rejected_segments", self.assistant_loss_mask),
        }
        if self.td_dpo_mode == "symmetric":
            if len(out["chosen_segments"]) != len(out["rejected_segments"]):
                raise RuntimeError(f"{row['pair_id']}: chosen/rejected segment counts differ for TD-DPO")
            chosen_weights: list[list[float]] = []
            rejected_weights: list[list[float]] = []
            for segment_index, (chosen, rejected) in enumerate(zip(out["chosen_segments"], out["rejected_segments"])):
                aligned = annotate_segment(
                    chosen["input_ids"].tolist(),
                    chosen["labels"].tolist(),
                    rejected["input_ids"].tolist(),
                    rejected["labels"].tolist(),
                    difference_weight=self.td_dpo_difference_weight,
                    shared_weight=self.td_dpo_shared_weight,
                )
                chosen_weights.append(
                    apply_visible_difference_multiplier(
                        aligned.chosen_weights,
                        aligned.chosen_categories,
                        chosen["td_dpo_visible_content_mask"].tolist(),
                        multiplier=self.td_dpo_visible_difference_multiplier,
                    )
                )
                rejected_weights.append(
                    apply_visible_difference_multiplier(
                        aligned.rejected_weights,
                        aligned.rejected_categories,
                        rejected["td_dpo_visible_content_mask"].tolist(),
                        multiplier=self.td_dpo_visible_difference_multiplier,
                    )
                )
            if self.td_dpo_normalize_branch_weights:
                chosen_weights = normalize_branch_weights(chosen_weights)
                rejected_weights = normalize_branch_weights(rejected_weights)
            for segment, weights in zip(out["chosen_segments"], chosen_weights):
                segment["td_dpo_weights"] = torch.tensor(weights, dtype=torch.float32)
            for segment, weights in zip(out["rejected_segments"], rejected_weights):
                segment["td_dpo_weights"] = torch.tensor(weights, dtype=torch.float32)
        if self.chosen_nll_mask is not None:
            if self.assistant_loss_mask == self.chosen_nll_mask:
                out["chosen_nll_labels"] = [
                    segment["labels"].clone() for segment in out["chosen_segments"]
                ]
            else:
                nll_chosen = tokenize_segments("chosen_segments", self.chosen_nll_mask)
                if len(nll_chosen) != len(out["chosen_segments"]):
                    raise AssertionError(f"{row.get('pair_id')}: chosen NLL segment count changed")
                nll_labels: list[torch.Tensor] = []
                for segment_index, (dpo_segment, nll_segment) in enumerate(zip(out["chosen_segments"], nll_chosen)):
                    if not torch.equal(dpo_segment["input_ids"], nll_segment["input_ids"]):
                        raise AssertionError(
                            f"{row.get('pair_id')}: chosen NLL rendering changed input IDs in segment {segment_index}"
                        )
                    nll_labels.append(nll_segment["labels"])
                out["chosen_nll_labels"] = nll_labels
        for field in ("reference_chosen_logps", "reference_rejected_logps"):
            if field in row:
                out[field] = torch.tensor(float(row[field]), dtype=torch.float32)
        return out


class PairCollator:
    def __init__(
        self,
        pad_token_id: int,
        *,
        tokenizer=None,
        max_length: int | None = None,
        assistant_loss_mask: str = "content_only",
        kto_enabled: bool = False,
        chat_rendering: str = LEGACY_CHAT_RENDERING,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_loss_mask = assistant_loss_mask
        self.kto_enabled = kto_enabled
        self.chat_rendering = chat_rendering
        if self.kto_enabled and (self.tokenizer is None or self.max_length is None):
            raise ValueError("KTO collation requires tokenizer and max_length.")

    @staticmethod
    def _pad(items: list[torch.Tensor], value: int | float) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(item.shape[0]) for item in items)
        values: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for item in items:
            padding = max_len - int(item.shape[0])
            if padding:
                item = torch.cat([item, torch.full((padding,), value, dtype=item.dtype)])
            values.append(item)
            masks.append(torch.cat([torch.ones(max_len - padding, dtype=torch.long), torch.zeros(padding, dtype=torch.long)]))
        return torch.stack(values), torch.stack(masks)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        def collate_side(side: str) -> dict[str, torch.Tensor]:
            input_ids: list[torch.Tensor] = []
            labels: list[torch.Tensor] = []
            chosen_nll_labels: list[torch.Tensor] = []
            td_dpo_weights: list[torch.Tensor] = []
            owners: list[int] = []
            for owner, item in enumerate(features):
                nll_segments = item.get("chosen_nll_labels") if side == "chosen" else None
                if nll_segments is not None and len(nll_segments) != len(item[f"{side}_segments"]):
                    raise RuntimeError(f"{item['pair_id']}: chosen NLL labels do not match chosen segments")
                for segment_index, segment in enumerate(item[f"{side}_segments"]):
                    input_ids.append(segment["input_ids"])
                    labels.append(segment["labels"])
                    if "td_dpo_weights" in segment:
                        td_dpo_weights.append(segment["td_dpo_weights"])
                    if nll_segments is not None:
                        chosen_nll_labels.append(nll_segments[segment_index])
                    owners.append(owner)
            padded_ids, attention = self._pad(input_ids, self.pad_token_id)
            padded_labels, _ = self._pad(labels, LABEL_PAD_TOKEN_ID)
            output = {
                f"{side}_input_ids": padded_ids,
                f"{side}_attention_mask": attention,
                f"{side}_labels": padded_labels,
                f"{side}_segment_owner": torch.tensor(owners, dtype=torch.long),
            }
            if chosen_nll_labels:
                padded_nll_labels, _ = self._pad(chosen_nll_labels, LABEL_PAD_TOKEN_ID)
                output["chosen_nll_labels"] = padded_nll_labels
            if td_dpo_weights:
                if len(td_dpo_weights) != len(input_ids):
                    raise RuntimeError("TD-DPO weights are missing from one or more assistant segments")
                padded_td_dpo_weights, _ = self._pad(td_dpo_weights, 0.0)
                output[f"{side}_td_dpo_weights"] = padded_td_dpo_weights
            return output

        batch: dict[str, Any] = {
            "pair_id": [item["pair_id"] for item in features],
        }
        batch.update(collate_side("chosen"))
        batch.update(collate_side("rejected"))
        if self.kto_enabled:
            # KTO estimates its detached reference point from completions
            # cyclically mismatched against other prompts in this *actual*
            # local minibatch.  Put all chosen segments first and all rejected
            # segments second, so two source pairs yield c0,c1,r0,r1: the
            # cyclic permutation cannot pair a completion back to its own
            # source pair solely because the two branches were adjacent.
            kto_segments: list[tuple[dict[str, Any], int]] = []
            for side in ("chosen", "rejected"):
                for owner, item in enumerate(features):
                    kto_segments.extend((segment, owner) for segment in item[f"{side}_segments"])
            if len(kto_segments) < 2:
                raise RuntimeError("KTO requires at least two assistant segments in each local minibatch.")
            kl_ids: list[torch.Tensor] = []
            kl_labels: list[torch.Tensor] = []
            identity_fallbacks = 0
            for index, (segment, owner) in enumerate(kto_segments):
                # Start at the next segment and rotate cyclically, exactly as
                # the usual KTO estimator does.  Prefer a completion from a
                # *different source pair* and only fall back to the original
                # completion when every cross-pair candidate would exceed our
                # fixed context budget.  This avoids silent truncation while
                # making the exceptional identity cases observable in logs.
                selected: tuple[list[int], list[int]] | None = None
                for offset in range(1, len(kto_segments)):
                    candidate, candidate_owner = kto_segments[(index + offset) % len(kto_segments)]
                    if candidate_owner == owner:
                        continue
                    messages = [*segment["kto_prompt_messages"], candidate["kto_completion_message"]]
                    input_ids, labels = render_sequence(
                        self.tokenizer,
                        messages,
                        assistant_loss_mask=self.assistant_loss_mask,
                        chat_rendering=self.chat_rendering,
                    )
                    if len(input_ids) <= self.max_length:
                        selected = (input_ids, labels)
                        break
                if selected is None:
                    messages = [*segment["kto_prompt_messages"], segment["kto_completion_message"]]
                    input_ids, labels = render_sequence(
                        self.tokenizer,
                        messages,
                        assistant_loss_mask=self.assistant_loss_mask,
                        chat_rendering=self.chat_rendering,
                    )
                    if len(input_ids) > self.max_length:
                        raise RuntimeError(
                            "KTO source prompt/completion sequence exceeds --max-length "
                            f"{self.max_length}; the prepared dataset itself is incompatible with this run."
                        )
                    selected = (input_ids, labels)
                    identity_fallbacks += 1
                input_ids, labels = selected
                kl_ids.append(torch.tensor(input_ids, dtype=torch.long))
                kl_labels.append(torch.tensor(labels, dtype=torch.long))
            padded_ids, attention = self._pad(kl_ids, self.pad_token_id)
            padded_labels, _ = self._pad(kl_labels, LABEL_PAD_TOKEN_ID)
            batch.update(
                {
                    "kto_kl_input_ids": padded_ids,
                    "kto_kl_attention_mask": attention,
                    "kto_kl_labels": padded_labels,
                    "kto_kl_identity_fallbacks": torch.tensor(identity_fallbacks, dtype=torch.long),
                }
            )
        for field in ("reference_chosen_logps", "reference_rejected_logps"):
            if all(field in item for item in features):
                batch[field] = torch.stack([item[field] for item in features])
        return batch


def selected_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gathered next-token log-probs and the corresponding label mask."""
    logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(LABEL_PAD_TOKEN_ID)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return token_logps, valid


def sequence_logps_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    td_dpo_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    token_logps, valid = selected_token_logps(logits, labels)
    if td_dpo_weights is None:
        return (token_logps * valid).sum(dim=-1)
    if td_dpo_weights.shape != labels.shape:
        raise RuntimeError("TD-DPO weights and labels have different shapes")
    weights = td_dpo_weights[:, 1:].to(device=token_logps.device, dtype=token_logps.dtype)
    if torch.any(weights[~valid] != 0) or torch.any(weights[valid] <= 0):
        raise RuntimeError("TD-DPO weights must be positive exactly at selected label positions")
    return (token_logps * weights).sum(dim=-1)


def sequence_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    td_dpo_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return sequence_logps_from_logits(outputs.logits, labels, td_dpo_weights)


def chosen_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Length-normalized NLL over all selected chosen assistant tokens."""
    token_logps, valid = selected_token_logps(logits, labels)
    token_count = valid.sum()
    if int(token_count.item()) <= 0:
        raise RuntimeError("Chosen NLL target contains no trainable assistant tokens.")
    return -(token_logps * valid).sum() / token_count, token_count


def pair_logps(model, batch: dict[str, Any], side: str) -> torch.Tensor:
    """Sum native per-turn log-probability sums back into one DPO pair value."""
    segment_logps = sequence_logps(
        model,
        batch[f"{side}_input_ids"],
        batch[f"{side}_attention_mask"],
        batch[f"{side}_labels"],
        batch.get(f"{side}_td_dpo_weights"),
    )
    owners = batch[f"{side}_segment_owner"]
    pair_count = int(owners.max().item()) + 1
    values = segment_logps.new_zeros((pair_count,))
    values.index_add_(0, owners, segment_logps)
    return values


def reference_sequence_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Score a sequence under the frozen base policy of the active LoRA model.

    KTO needs both ordinary and cyclically mismatched reference scores.  Using
    PEFT's disabled-adapter context computes those exactly without loading a
    second 27B model or relying on cache values tied to a different minibatch.
    """
    adapter_model = model.module if hasattr(model, "module") and hasattr(model.module, "disable_adapter") else model
    if not hasattr(adapter_model, "disable_adapter"):
        raise RuntimeError("KTO requires a PEFT model exposing disable_adapter().")
    with torch.no_grad(), adapter_model.disable_adapter():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return sequence_logps_from_logits(outputs.logits, labels)


def chunked_sequence_logps(
    scorer,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Score a detached KTO pseudo-batch in bounded forwards.

    KTO's cyclic reference-point estimator is detached, so its scores can be
    concatenated after independent forwards. This retains every pseudo-sequence
    and their mean while bounding the temporary fp32 vocabulary softmax.
    """
    if chunk_size <= 0 or chunk_size >= int(input_ids.shape[0]):
        return scorer(model, input_ids, attention_mask, labels)
    values: list[torch.Tensor] = []
    for start in range(0, int(input_ids.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(input_ids.shape[0]))
        values.append(
            scorer(
                model,
                input_ids[start:stop],
                attention_mask[start:stop],
                labels[start:stop],
            )
        )
    return torch.cat(values, dim=0)


def pair_scored_token_counts(batch: dict[str, Any], side: str) -> torch.Tensor:
    """Scored assistant tokens per DPO pair, matching ``pair_logps`` aggregation.

    The DPO sum runs over next-token predictions, so this counts exactly the
    label positions that contribute to it and aggregates multi-turn segments
    back into their owning pair.
    """
    shifted_labels = batch[f"{side}_labels"][:, 1:]
    per_segment = shifted_labels.ne(LABEL_PAD_TOKEN_ID).sum(dim=-1).to(torch.float32)
    owners = batch[f"{side}_segment_owner"]
    pair_count = int(owners.max().item()) + 1
    counts = per_segment.new_zeros((pair_count,))
    counts.index_add_(0, owners, per_segment)
    # Preparation rejects pairs with no trainable assistant tokens, so this
    # clamp only guards against a divide-by-zero on an unexpected batch.
    return counts.clamp_min(1.0)


class AdapterSnapshotCallback(TrainerCallback):
    """Write adapter-only snapshots at explicit global steps.

    Deliberately not ``control.should_save``: a full Trainer checkpoint also
    stores Adam's two fp32 moments over every LoRA parameter, which is several
    GB apiece and would dominate disk once early checkpoints are dense.  These
    snapshots exist to be merged and evaluated, never resumed from, so the
    optimizer state is pure overhead.  Regular resumable checkpoints continue on
    the normal ``--save-steps`` cadence.
    """

    def __init__(self, steps: Iterable[int], snapshot_dir: Path, *, synchronize: bool = True) -> None:
        self.steps = {int(step) for step in steps if int(step) > 0}
        self.snapshot_dir = Path(snapshot_dir)
        self.synchronize = synchronize

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[no-untyped-def]
        step = int(state.global_step)
        if step not in self.steps:
            return control
        if state.is_world_process_zero:
            if model is None:
                raise RuntimeError("Rank zero received no model for an adapter snapshot.")
            target = self.snapshot_dir / f"adapter-{step:06d}"
            target.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            model.save_pretrained(str(target))
            print(
                f"[snapshot] adapter-only snapshot at step {step}: {target} "
                f"elapsed_s={time.perf_counter() - started:.3f}",
                flush=True,
            )
        if self.synchronize:
            distributed_barrier()
        return control


def parse_adapter_snapshot_steps(raw: str) -> list[int]:
    steps: list[int] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError as exc:
            raise ValueError(f"Invalid --adapter-snapshot-steps entry {chunk!r}; expected integers.") from exc
        if value <= 0:
            raise ValueError(f"--adapter-snapshot-steps entries must be positive, got {value}")
        steps.append(value)
    return sorted(set(steps))


def pair_logps_from_logits(logits: torch.Tensor, batch: dict[str, Any], side: str) -> torch.Tensor:
    """Aggregate already-computed per-segment logits into DPO pair log-probs."""
    segment_logps = sequence_logps_from_logits(
        logits,
        batch[f"{side}_labels"],
        batch.get(f"{side}_td_dpo_weights"),
    )
    owners = batch[f"{side}_segment_owner"]
    pair_count = int(owners.max().item()) + 1
    values = segment_logps.new_zeros((pair_count,))
    values.index_add_(0, owners, segment_logps)
    return values


class AssistantMaskDPOTrainer(Trainer):
    """Minimal DPO Trainer for arbitrary assistant spans in a chat trajectory."""

    def __init__(
        self,
        *args,
        beta: float,
        preference_objective: str = "dpo",
        kto_desirable_weight: float = 1.0,
        kto_undesirable_weight: float = 1.0,
        kto_kl_score_batch_size: int = 0,
        chosen_nll_coef: float = 0.0,
        chosen_nll_scale_mode: str = "fixed",
        chosen_nll_scale_ema: float = 0.99,
        length_normalize_dpo: bool = False,
        dpop_penalty_weight: float = 0.0,
        synchronize_checkpoints: bool = True,
        step_timing_dir: str | None = None,
        consumed_pairs_dir: str | None = None,
        detailed_timing_recorder: DetailedTimingRecorder | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.preference_objective = preference_objective
        self.kto_desirable_weight = kto_desirable_weight
        self.kto_undesirable_weight = kto_undesirable_weight
        self.kto_kl_score_batch_size = kto_kl_score_batch_size
        self.chosen_nll_coef = chosen_nll_coef
        self.chosen_nll_scale_mode = chosen_nll_scale_mode
        self.chosen_nll_scale_ema = chosen_nll_scale_ema
        self.length_normalize_dpo = length_normalize_dpo
        self.dpop_penalty_weight = dpop_penalty_weight
        self.synchronize_checkpoints = synchronize_checkpoints
        self.step_timing_dir = Path(step_timing_dir) if step_timing_dir else None
        self.consumed_pairs_dir = Path(consumed_pairs_dir) if consumed_pairs_dir else None
        if self.consumed_pairs_dir is not None:
            self.consumed_pairs_dir.mkdir(parents=True, exist_ok=True)
        self.detailed_timing_recorder = detailed_timing_recorder
        self._last_dpo_metrics: dict[str, float] = {}
        self._metric_accumulator: list[dict[str, float]] = []
        # The NLL scale starts unsaturated so the regularizer has full strength
        # while the DPO term is still doing work.
        self._nll_scale: float = 1.0

    def training_step(self, model, inputs, *args, **kwargs):  # type: ignore[override]
        """Record synchronized full forward/backward timing for each local pair."""
        pair_ids = list(inputs.get("pair_id", []))
        if self.consumed_pairs_dir is not None and pair_ids:
            rank = int(os.environ.get("RANK", "0"))
            path = self.consumed_pairs_dir / f"rank_{rank:03d}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "global_step_before_update": int(self.state.global_step),
                            "rank": rank,
                            "pair_ids": pair_ids,
                        }
                    )
                    + "\n"
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        loss = super().training_step(model, inputs, *args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if self.step_timing_dir is not None:
            self.step_timing_dir.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            path = self.step_timing_dir / f"rank_{rank:03d}.jsonl"
            record = {
                "global_step_before_update": int(self.state.global_step),
                "rank": rank,
                "pair_ids": pair_ids,
                "local_batch_size": len(pair_ids),
                "elapsed_seconds_forward_backward": elapsed,
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        if self.detailed_timing_recorder is not None:
            self.detailed_timing_recorder.record_inner_batch(pair_ids=pair_ids, elapsed_seconds=elapsed)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):  # type: ignore[override]
        """Evaluate custom DPO batches through ``compute_loss``.

        The stock ``Trainer`` only selects its loss path when it recognises a
        conventional ``labels`` field.  Our batches instead contain paired,
        assistant-masked chosen/rejected sequences, so its fallback tries to
        call Qwen directly without ``input_ids``.  Validation must use exactly
        the same DPO objective as training and does not need logits or labels.
        """
        required = {"chosen_input_ids", "rejected_input_ids"}
        if self.preference_objective == "kto":
            required.update({"kto_kl_input_ids", "kto_kl_labels"})
        else:
            required.update({"reference_chosen_logps", "reference_rejected_logps"})
        if not required.issubset(inputs):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.mean().detach(), None, None

    def _update_nll_scale(self, logits: torch.Tensor, *, is_training: bool) -> float:
        """EMA of the DPO saturation factor sigma(-x), shared across ranks.

        sigma(-x) is exactly the factor multiplying every per-token DPO
        gradient, so tying the NLL coefficient to it makes the regularizer fade
        out precisely as the contrastive term stops contributing, instead of
        becoming the entire objective once DPO saturates.

        ``is_training`` must be captured by the caller *before* entering any
        ``torch.no_grad`` region: querying grad mode from inside one always
        reports False and would silently skip the cross-rank reduction.
        """
        with torch.no_grad():
            # Keep the scheduler independent of autocast's bf16 underflow
            # threshold.  The DPO computation retains its configured dtype;
            # this detached scalar is only used to scale the auxiliary term.
            instantaneous = torch.sigmoid(-logits.detach().float()).mean()
            # Only reduce on the training path. Evaluation can have uneven
            # per-rank batch counts, and an unmatched collective would deadlock.
            if is_training and torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(instantaneous, op=torch.distributed.ReduceOp.SUM)
                instantaneous = instantaneous / float(torch.distributed.get_world_size())
            momentum = self.chosen_nll_scale_ema
            self._nll_scale = momentum * self._nll_scale + (1.0 - momentum) * float(instantaneous)
        return self._nll_scale

    def _save_checkpoint(self, model, trial, *args, **kwargs):  # type: ignore[override]
        """Persist the NLL scale alongside the checkpoint.

        The EMA is trainer state, not model state. Without this, an auto-resumed
        job restarts the scale at its initial value and delivers a burst of
        full-strength regularization mid-run - exactly what the schedule exists
        to prevent, and invisible unless nll/scale is being watched.
        """
        step = int(self.state.global_step)
        started = time.perf_counter()
        result = super()._save_checkpoint(model, trial, *args, **kwargs)
        # Trainer writes shared state only on rank zero but lets nonzero ranks
        # return quickly after their rank-local RNG writes. Keep every rank out
        # of the next DDP collective until that shared checkpoint is durable.
        if self.synchronize_checkpoints:
            distributed_barrier()
        if self.is_world_process_zero():
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{step}"
            if checkpoint_dir.is_dir():
                (checkpoint_dir / "dpo_trainer_state.json").write_text(
                    json.dumps({"nll_scale": float(self._nll_scale)}) + "\n",
                    encoding="utf-8",
                )
                # The marker is written only after Trainer's save has returned
                # and the custom EMA state is present. The launcher can then
                # reject interrupted checkpoint directories during auto-resume.
                (checkpoint_dir / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")
                print(
                    f"[checkpoint] complete step={step} elapsed_s={time.perf_counter() - started:.3f} "
                    f"dir={checkpoint_dir}",
                    flush=True,
                )
        if self.synchronize_checkpoints:
            distributed_barrier()
        return result

    def _compute_kto_loss(self, model, inputs, *, return_outputs: bool, is_training: bool):
        """KTO over binary chosen/rejected assistant turns from the paired view.

        The KTO reference point is the detached, clamped mean policy/reference
        log-ratio on the cyclically mismatched assistant turns that the
        collator constructed from this *actual* local minibatch.  This matches
        the standard KTO estimator while avoiding a second resident 27B model.
        """
        # Calculate all detached quantities *before* either policy forward
        # creates a gradient activation graph.  On long Qwen trajectories the
        # reference scorer temporarily materializes a large fp32 log-softmax
        # workspace.  The prior ordering retained chosen and rejected policy
        # graphs underneath that workspace and made otherwise-valid local
        # microbatch-2 KTO runs OOM.  This is only a lifetime change: every
        # scalar below is detached in standard KTO, so it cannot alter the
        # objective or its gradients.
        reference_chosen = reference_sequence_logps(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_labels"],
        )
        reference_rejected = reference_sequence_logps(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_labels"],
        )
        # KTO explicitly stops gradients through its reference-point/KL
        # estimate.  Compute the policy half under no_grad too: retaining a
        # third full 27B activation graph would consume memory without ever
        # contributing to the update.
        with torch.no_grad():
            kto_kl_logps = chunked_sequence_logps(
                sequence_logps,
                model,
                inputs["kto_kl_input_ids"],
                inputs["kto_kl_attention_mask"],
                inputs["kto_kl_labels"],
                chunk_size=self.kto_kl_score_batch_size,
            )
            reference_kto_kl = chunked_sequence_logps(
                reference_sequence_logps,
                model,
                inputs["kto_kl_input_ids"],
                inputs["kto_kl_attention_mask"],
                inputs["kto_kl_labels"],
                chunk_size=self.kto_kl_score_batch_size,
            )
            reference_point = (kto_kl_logps - reference_kto_kl).mean().clamp_min(0.0)
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            use_cache=False,
        )
        # Unlike DPO, KTO is a binary objective over individual prompt /
        # completion examples.  Keep prepared dialogue turns separate here;
        # summing a trajectory back to one pair would make the matched reward
        # population disagree with the cyclically mismatched KTO population.
        chosen_logps = sequence_logps_from_logits(chosen_outputs.logits, inputs["chosen_labels"])
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            use_cache=False,
        )
        rejected_logps = sequence_logps_from_logits(rejected_outputs.logits, inputs["rejected_labels"])
        chosen_logratios = chosen_logps - reference_chosen
        rejected_logratios = rejected_logps - reference_rejected
        loss, desirable_loss, undesirable_loss = kto_loss_from_logratios(
            chosen_logratios,
            rejected_logratios,
            reference_point=reference_point,
            beta=self.beta,
            desirable_weight=self.kto_desirable_weight,
            undesirable_weight=self.kto_undesirable_weight,
        )
        with torch.no_grad():
            self._last_dpo_metrics = {
                "kto/loss": float(loss.detach().cpu()),
                "kto/desirable_loss": float(desirable_loss.detach().cpu()),
                "kto/undesirable_loss": float(undesirable_loss.detach().cpu()),
                "kto/reference_point": float(reference_point.detach().cpu()),
                "kto/desirable_logratio": float(chosen_logratios.mean().detach().cpu()),
                "kto/undesirable_logratio": float(rejected_logratios.mean().detach().cpu()),
                "kto/kl_logratio": float((kto_kl_logps - reference_kto_kl).mean().detach().cpu()),
                "kto/kl_identity_fallbacks": float(inputs["kto_kl_identity_fallbacks"].detach().cpu()),
                "kto/kl_score_batch_size": float(self.kto_kl_score_batch_size),
                "kto/desirable_weight": float(self.kto_desirable_weight),
                "kto/undesirable_weight": float(self.kto_undesirable_weight),
                "kto/desirable_segments": float(chosen_logps.numel()),
                "kto/undesirable_segments": float(rejected_logps.numel()),
                "objective/total_loss": float(loss.detach().cpu()),
            }
            if is_training:
                self._metric_accumulator.append(dict(self._last_dpo_metrics))
        if return_outputs:
            return loss, {"chosen_logps": chosen_logps, "rejected_logps": rejected_logps}
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        # Capture grad mode before any helper enters a torch.no_grad context.
        is_training = torch.is_grad_enabled()
        if self.preference_objective == "kto":
            return self._compute_kto_loss(model, inputs, return_outputs=return_outputs, is_training=is_training)
        if "reference_chosen_logps" not in inputs or "reference_rejected_logps" not in inputs:
            raise RuntimeError("Training requires precomputed chosen/rejected reference log-probabilities.")
        # Capture grad mode up front: everything below that needs to distinguish
        # the training path from evaluation runs inside a torch.no_grad region,
        # where torch.is_grad_enabled() would always report False.
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            use_cache=False,
        )
        chosen_logps = pair_logps_from_logits(chosen_outputs.logits, inputs, "chosen")
        rejected_logps = pair_logps(model, inputs, "rejected")
        reference_chosen = inputs["reference_chosen_logps"].to(chosen_logps.device, dtype=chosen_logps.dtype)
        reference_rejected = inputs["reference_rejected_logps"].to(rejected_logps.device, dtype=rejected_logps.dtype)
        # Reference values are stored as sums, so length normalization divides
        # policy and reference by the same per-pair token counts.  Doing it here
        # rather than inside pair_logps keeps the reference-alignment guard and
        # the cached *_with_reference.jsonl views valid without recomputation.
        chosen_score, rejected_score = chosen_logps, rejected_logps
        reference_chosen_score, reference_rejected_score = reference_chosen, reference_rejected
        if self.length_normalize_dpo:
            chosen_tokens = pair_scored_token_counts(inputs, "chosen").to(chosen_logps.device, dtype=chosen_logps.dtype)
            rejected_tokens = pair_scored_token_counts(inputs, "rejected").to(rejected_logps.device, dtype=rejected_logps.dtype)
            chosen_score = chosen_logps / chosen_tokens
            rejected_score = rejected_logps / rejected_tokens
            reference_chosen_score = reference_chosen / chosen_tokens
            reference_rejected_score = reference_rejected / rejected_tokens
        preference_logit = (chosen_score - rejected_score) - (reference_chosen_score - reference_rejected_score)
        dpop_penalty = None
        if self.dpop_penalty_weight:
            # DPO-Positive: prevent the chosen response from becoming less
            # likely than under the frozen reference.  This uses the same
            # sequence log-probabilities and mask as the DPO objective.
            dpop_penalty = (reference_chosen_score - chosen_score).clamp_min(0)
            preference_logit = preference_logit - self.dpop_penalty_weight * dpop_penalty
        logits = self.beta * preference_logit
        dpo_loss = -F.logsigmoid(logits).mean()
        chosen_nll = None
        chosen_nll_token_count = None
        effective_nll_coef = self.chosen_nll_coef
        if self.chosen_nll_coef:
            if "chosen_nll_labels" not in inputs:
                raise RuntimeError("Chosen NLL is enabled but the batch lacks chosen_nll_labels.")
            if self.chosen_nll_scale_mode == "dpo_saturation":
                effective_nll_coef = self.chosen_nll_coef * self._update_nll_scale(
                    logits, is_training=is_training
                )
            chosen_nll, chosen_nll_token_count = chosen_nll_from_logits(
                chosen_outputs.logits,
                inputs["chosen_nll_labels"],
            )
            loss = dpo_loss + effective_nll_coef * chosen_nll
        else:
            loss = dpo_loss
        with torch.no_grad():
            self._last_dpo_metrics = {
                "dpo/loss": float(dpo_loss.detach().cpu()),
                "dpo/chosen_logps": float(chosen_logps.mean().detach().cpu()),
                "dpo/rejected_logps": float(rejected_logps.mean().detach().cpu()),
                "dpo/reward_margin": float(logits.mean().detach().cpu()),
                "dpo/preference_accuracy": float((logits > 0).float().mean().detach().cpu()),
                "objective/total_loss": float(loss.detach().cpu()),
            }
            # The saturation factor sigma(-x) multiplies every per-token DPO
            # gradient, and its *arithmetic* mean is what the NLL schedule
            # consumes.  Log that exact quantity so the dashboard can validate
            # the EMA driving the coefficient.  A geometric-style mean of
            # log-sigma would read as fully saturated even when a small
            # fraction of pairs still carries essentially all of the gradient.
            logits_fp32 = logits.float()
            self._last_dpo_metrics["dpo/saturation_sigma"] = float(
                torch.sigmoid(-logits_fp32).mean().detach().cpu()
            )
            # Numerically stable log of that same arithmetic mean.  The linear
            # value eventually underflows float32, so this keeps resolution in
            # the deeply saturated regime.
            # Note the cross-microbatch average of this field is geometric;
            # dpo/saturation_sigma is the one to read for gradient magnitude.
            log_sigma = F.logsigmoid(-logits_fp32)
            self._last_dpo_metrics["dpo/log_mean_saturation_sigma"] = float(
                (torch.logsumexp(log_sigma, dim=0) - math.log(log_sigma.numel())).detach().cpu()
            )
            if self.length_normalize_dpo:
                self._last_dpo_metrics["dpo/length_normalized"] = 1.0
            if chosen_nll is not None and chosen_nll_token_count is not None:
                self._last_dpo_metrics.update(
                    {
                        "nll/chosen": float(chosen_nll.detach().cpu()),
                        "nll/coefficient": float(self.chosen_nll_coef),
                        "nll/effective_coefficient": float(effective_nll_coef),
                        "nll/scale": float(self._nll_scale),
                        "nll/chosen_tokens": float(chosen_nll_token_count.detach().cpu()),
                    }
                )
            if dpop_penalty is not None:
                self._last_dpo_metrics.update(
                    {
                        "dpop/penalty": float(dpop_penalty.mean().detach().cpu()),
                        "dpop/penalty_weight": float(self.dpop_penalty_weight),
                        "dpop/fraction_active": float((dpop_penalty > 0).float().mean().detach().cpu()),
                    }
                )
            # Accumulate last, so every metric above is included in the average.
            if is_training:
                self._metric_accumulator.append(dict(self._last_dpo_metrics))
        if return_outputs:
            return loss, {"chosen_logps": chosen_logps, "rejected_logps": rejected_logps}
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:  # type: ignore[override]
        logs = dict(logs)
        is_eval = any(key.startswith("eval_") for key in logs)
        # Training metrics average over every microbatch *on this rank* since
        # the last log.  A single-pair snapshot made per-pair statistics
        # (preference accuracy, DPOP activity) uninterpretable at microbatch
        # size 1.  This is not a global-batch statistic: only rank zero emits
        # logs, so it covers that rank's accumulation steps, not all ranks.
        # The one quantity that must agree across ranks - the NLL scale - is
        # all-reduced separately in _update_nll_scale.
        source = self._last_dpo_metrics
        if not is_eval and self._metric_accumulator:
            keys = {key for record in self._metric_accumulator for key in record}
            source = {
                key: sum(record[key] for record in self._metric_accumulator if key in record)
                / max(1, sum(1 for record in self._metric_accumulator if key in record))
                for key in keys
            }
            self._metric_accumulator = []
        if source:
            prefix = "eval_" if is_eval else ""
            logs.update({prefix + key.replace("/", "_"): value for key, value in source.items()})
        try:
            super().log(logs, start_time=start_time)
        except TypeError:  # transformers compatibility
            super().log(logs)


def build_model(args: argparse.Namespace, *, with_lora: bool):
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.config.use_cache = False
    if not with_lora:
        return model
    targets: Any = args.target_modules
    if isinstance(targets, list) and len(targets) == 1:
        targets = targets[0]
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def precompute_reference(args: argparse.Namespace) -> int:
    if args.output_path is None:
        raise ValueError("--output-path is required for precompute-reference mode")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    rows = load_rows(Path(args.data_jsonl), args.max_examples, assistant_loss_mask=args.assistant_loss_mask)
    selected = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = PreparedPairDataset(
        selected,
        tokenizer,
        args.max_length,
        args.assistant_loss_mask,
        td_dpo_mode=args.td_dpo_mode,
        td_dpo_difference_weight=args.td_dpo_difference_weight,
        td_dpo_shared_weight=args.td_dpo_shared_weight,
        td_dpo_normalize_branch_weights=args.td_dpo_normalize_branch_weights,
        td_dpo_visible_difference_multiplier=args.td_dpo_visible_difference_multiplier,
        chat_rendering=args.chat_rendering,
    )
    collator = PairCollator(tokenizer.pad_token_id)
    # The frozen reference is the initial policy. Build it with the same
    # zero-initialized LoRA wrapper used by training, rather than the bare base
    # module, so summed log-probabilities use the same numerical forward path.
    model = build_model(args, with_lora=True).eval()
    if not torch.cuda.is_available():
        raise RuntimeError("Reference precomputation requires a CUDA GPU.")
    model.to("cuda")
    device = next(model.parameters()).device
    output: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), args.precompute_batch_size):
            features = [dataset[i] for i in range(start, min(start + args.precompute_batch_size, len(dataset)))]
            batch = collator(features)
            device_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            chosen = pair_logps(model, device_batch, "chosen")
            rejected = pair_logps(model, device_batch, "rejected")
            for pair_id, chosen_value, rejected_value in zip(batch["pair_id"], chosen.tolist(), rejected.tolist()):
                output.append(
                    {
                        "pair_id": pair_id,
                        "reference_chosen_logps": float(chosen_value),
                        "reference_rejected_logps": float(rejected_value),
                        "assistant_loss_mask": args.assistant_loss_mask,
                        **chat_rendering_metadata(args),
                        **td_dpo_reference_metadata(args),
                    }
                )
            if (start // args.precompute_batch_size + 1) % 25 == 0:
                print(f"[reference] completed={len(output)}/{len(dataset)}", flush=True)
    write_jsonl(Path(args.output_path), output, args.overwrite)
    print(f"[done] reference shard rows={len(output)} output={args.output_path}")
    return 0


def attach_reference(args: argparse.Namespace) -> int:
    if args.output_path is None:
        raise ValueError("--output-path is required for attach-reference mode")
    if not args.reference_jsonl:
        raise ValueError("--attach-reference requires one or more --reference-jsonl files")
    rows = load_rows(Path(args.data_jsonl), args.max_examples, assistant_loss_mask=args.assistant_loss_mask)
    values: dict[str, dict[str, float]] = {}
    for reference_path in args.reference_jsonl:
        for item in iter_jsonl(Path(reference_path)):
            pair_id = str(item.get("pair_id") or "")
            if not pair_id or pair_id in values:
                raise ValueError(f"Duplicate or missing reference pair_id in {reference_path}: {pair_id!r}")
            reference_mask = item.get("assistant_loss_mask")
            if reference_mask is not None and reference_mask != args.assistant_loss_mask:
                raise ValueError(
                    f"{reference_path}: reference pair {pair_id} uses assistant_loss_mask="
                    f"{reference_mask!r}, expected {args.assistant_loss_mask!r}"
                )
            validate_chat_rendering_reference_metadata(item, args, source=str(reference_path))
            validate_td_dpo_reference_metadata(item, args, source=str(reference_path))
            values[pair_id] = {
                "reference_chosen_logps": float(item["reference_chosen_logps"]),
                "reference_rejected_logps": float(item["reference_rejected_logps"]),
            }
    missing = [str(row.get("pair_id")) for row in rows if str(row.get("pair_id")) not in values]
    extras = sorted(set(values) - {str(row.get("pair_id")) for row in rows})
    if missing or extras:
        raise RuntimeError(f"Reference merge mismatch: missing={len(missing)} extras={len(extras)}")
    # Carry the scoring contract into the attached view as well as checking it
    # above.  This makes the immutable training JSONL self-describing during
    # later audits, rather than requiring the separate shard files.
    objective_metadata = {
        "assistant_loss_mask": args.assistant_loss_mask,
        **chat_rendering_metadata(args),
        **td_dpo_reference_metadata(args),
    }
    output = [{**row, **values[str(row["pair_id"])], **objective_metadata} for row in rows]
    write_jsonl(Path(args.output_path), output, args.overwrite)
    print(f"[done] attached reference values to {len(output)} rows: {args.output_path}")
    return 0


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    if args.output_dir is None:
        raise ValueError("--output-dir is required for train mode")
    init_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": False,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.microbatch_size,
        # KTO's cyclic reference-point estimate needs at least two assistant
        # segments in a local batch.  A single source pair already supplies its
        # chosen and rejected segments, but matching the train microbatch here
        # keeps evaluation numerically representative as well.
        "per_device_eval_batch_size": args.microbatch_size if args.preference_objective == "kto" else 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "ddp_timeout": args.ddp_timeout_seconds,
        "evaluation_strategy": "steps" if args.eval_jsonl else "no",
        "eval_steps": args.eval_steps or args.save_steps,
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False,
        "seed": args.seed,
    }
    if "evaluation_strategy" not in init_params and "eval_strategy" in init_params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in init_params})


def verify_reference_alignment(model, dataset: PreparedPairDataset, collator: PairCollator, tolerance: float, count: int) -> None:
    if count <= 0:
        return
    model.eval()
    device = next(model.parameters()).device
    max_error = 0.0
    with torch.inference_mode():
        for index in range(min(count, len(dataset))):
            batch = collator([dataset[index]])
            if "reference_chosen_logps" not in batch:
                raise RuntimeError("Prepared training data is missing reference log-probabilities.")
            tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            chosen = pair_logps(model, tensors, "chosen")
            rejected = pair_logps(model, tensors, "rejected")
            max_error = max(
                max_error,
                abs(float(chosen.item()) - float(tensors["reference_chosen_logps"].item())),
                abs(float(rejected.item()) - float(tensors["reference_rejected_logps"].item())),
            )
    model.train()
    print(f"[reference-check] examples={min(count, len(dataset))} max_abs_error={max_error:.6f}")
    if max_error > tolerance:
        raise RuntimeError(
            f"Initial policy/reference mismatch {max_error:.6f} exceeds tolerance {tolerance}. "
            "Do not train until the rendering/reference path is reconciled."
        )


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def train(args: argparse.Namespace) -> int:
    if args.final_adapter_dir is None:
        raise ValueError("--final-adapter-dir is required for train mode")
    if args.chosen_nll_coef < 0:
        raise ValueError("--chosen-nll-coef must be non-negative")
    if args.dpop_penalty_weight < 0:
        raise ValueError("--dpop-penalty-weight must be non-negative")
    if (
        args.td_dpo_difference_weight <= 0
        or args.td_dpo_shared_weight <= 0
        or args.td_dpo_visible_difference_multiplier <= 0
    ):
        raise ValueError("TD-DPO difference/shared/visible-difference weights must all be positive")
    if args.td_dpo_mode != "none" and args.length_normalize_dpo:
        raise ValueError("TD-DPO branch normalization and --length-normalize-dpo are separate objectives; do not combine them.")
    if args.chosen_nll_coef > 0 and args.dpop_penalty_weight > 0:
        raise ValueError("--chosen-nll-coef and --dpop-penalty-weight are mutually exclusive.")
    if not 0.0 <= args.chosen_nll_scale_ema < 1.0:
        raise ValueError("--chosen-nll-scale-ema must be in [0, 1)")
    if args.chosen_nll_scale_mode != "fixed" and args.chosen_nll_coef <= 0:
        raise ValueError("--chosen-nll-scale-mode only applies when --chosen-nll-coef is positive.")
    if args.preference_objective == "kto":
        if args.beta <= 0:
            raise ValueError("KTO requires --beta to be positive.")
        if args.kto_desirable_weight <= 0 or args.kto_undesirable_weight <= 0:
            raise ValueError("KTO desirable/undesirable weights must both be positive.")
        if args.microbatch_size < 2:
            raise ValueError(
                "KTO requires --microbatch-size >= 2 so its cyclic KL/reference-point estimate mixes "
                "completions across source pairs rather than only across the two branches of one pair."
            )
        if args.kto_kl_score_batch_size < 0:
            raise ValueError("KTO --kto-kl-score-batch-size must be non-negative.")
        incompatible = {
            "--chosen-nll-coef": args.chosen_nll_coef > 0,
            "--dpop-penalty-weight": args.dpop_penalty_weight > 0,
            "--length-normalize-dpo": args.length_normalize_dpo,
            "--td-dpo-mode": args.td_dpo_mode != "none",
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError("KTO is a standalone preference objective; do not combine it with " + ", ".join(enabled))
    snapshot_steps = parse_adapter_snapshot_steps(args.adapter_snapshot_steps)
    snapshot_dir = (
        Path(args.adapter_snapshot_dir)
        if args.adapter_snapshot_dir
        else Path(args.output_dir).parent / "adapter_snapshots"
    )
    detailed_timing = DetailedTimingRecorder(args.detailed_timing_dir)
    detailed_timing.event("train_function_begin")
    rows = load_rows(Path(args.data_jsonl), args.max_examples, assistant_loss_mask=args.assistant_loss_mask)
    if args.preference_objective == "dpo":
        if not all("reference_chosen_logps" in row and "reference_rejected_logps" in row for row in rows):
            raise RuntimeError("DPO train JSONL must be produced by --mode attach-reference.")
        for row in rows:
            validate_chat_rendering_reference_metadata(row, args, source=f"train pair {row.get('pair_id')}")
            validate_td_dpo_reference_metadata(row, args, source=f"train pair {row.get('pair_id')}")
    eval_rows = load_rows(Path(args.eval_jsonl), assistant_loss_mask=args.assistant_loss_mask) if args.eval_jsonl else None
    if args.preference_objective == "dpo":
        if eval_rows and not all("reference_chosen_logps" in row and "reference_rejected_logps" in row for row in eval_rows):
            raise RuntimeError("DPO validation JSONL must also have attached reference log-probabilities.")
        for row in eval_rows or []:
            validate_chat_rendering_reference_metadata(row, args, source=f"validation pair {row.get('pair_id')}")
            validate_td_dpo_reference_metadata(row, args, source=f"validation pair {row.get('pair_id')}")
    set_seed(args.seed)
    detailed_timing.event("data_loaded", train_rows=len(rows), eval_rows=len(eval_rows or []))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = PreparedPairDataset(
        rows,
        tokenizer,
        args.max_length,
        args.assistant_loss_mask,
        chosen_nll_mask=args.chosen_nll_mask if args.chosen_nll_coef > 0 else None,
        td_dpo_mode=args.td_dpo_mode,
        td_dpo_difference_weight=args.td_dpo_difference_weight,
        td_dpo_shared_weight=args.td_dpo_shared_weight,
        td_dpo_normalize_branch_weights=args.td_dpo_normalize_branch_weights,
        td_dpo_visible_difference_multiplier=args.td_dpo_visible_difference_multiplier,
        kto_enabled=args.preference_objective == "kto",
        chat_rendering=args.chat_rendering,
    )
    eval_dataset = (
        PreparedPairDataset(
            eval_rows,
            tokenizer,
            args.max_length,
            args.assistant_loss_mask,
            chosen_nll_mask=args.chosen_nll_mask if args.chosen_nll_coef > 0 else None,
            td_dpo_mode=args.td_dpo_mode,
            td_dpo_difference_weight=args.td_dpo_difference_weight,
            td_dpo_shared_weight=args.td_dpo_shared_weight,
            td_dpo_normalize_branch_weights=args.td_dpo_normalize_branch_weights,
            td_dpo_visible_difference_multiplier=args.td_dpo_visible_difference_multiplier,
            kto_enabled=args.preference_objective == "kto",
            chat_rendering=args.chat_rendering,
        )
        if eval_rows
        else None
    )
    collator = PairCollator(
        tokenizer.pad_token_id,
        tokenizer=tokenizer,
        max_length=args.max_length,
        assistant_loss_mask=args.assistant_loss_mask,
        kto_enabled=args.preference_objective == "kto",
        chat_rendering=args.chat_rendering,
    )
    detailed_timing.event("tokenizer_and_datasets_ready")
    model = build_model(args, with_lora=True)
    detailed_timing.event("model_loaded")
    if not torch.cuda.is_available():
        raise RuntimeError("DPO training requires a CUDA GPU.")
    # Reference precomputation runs this same zero-initialized LoRA model on
    # CUDA.  Move before the alignment guard as well: otherwise the guard is
    # both extremely slow and compares CPU numerics against GPU numerics.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    model.to(torch.device("cuda", local_rank))
    detailed_timing.event("model_moved_to_cuda", local_rank=local_rank)
    if args.preference_objective == "dpo":
        verify_reference_alignment(model, train_dataset, collator, args.reference_tolerance, args.reference_check_examples)
    else:
        print(
            "[kto] reference policy is the same SDF base with its trainable LoRA adapter disabled; "
            "matched and cyclic-mismatch scores are calculated online.",
            flush=True,
        )
    detailed_timing.event("reference_alignment_complete")
    callbacks: list[TrainerCallback] = []
    if detailed_timing.enabled:
        callbacks.append(DetailedStepTimingCallback(detailed_timing))
    if snapshot_steps:
        callbacks.append(
            AdapterSnapshotCallback(snapshot_steps, snapshot_dir, synchronize=args.synchronize_checkpoints)
        )
        print(f"[snapshot] adapter-only snapshot steps {snapshot_steps} -> {snapshot_dir}", flush=True)
    trainer = AssistantMaskDPOTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        beta=args.beta,
        preference_objective=args.preference_objective,
        kto_desirable_weight=args.kto_desirable_weight,
        kto_undesirable_weight=args.kto_undesirable_weight,
        kto_kl_score_batch_size=args.kto_kl_score_batch_size,
        chosen_nll_coef=args.chosen_nll_coef,
        chosen_nll_scale_mode=args.chosen_nll_scale_mode,
        chosen_nll_scale_ema=args.chosen_nll_scale_ema,
        length_normalize_dpo=args.length_normalize_dpo,
        dpop_penalty_weight=args.dpop_penalty_weight,
        synchronize_checkpoints=args.synchronize_checkpoints,
        step_timing_dir=args.step_timing_dir,
        consumed_pairs_dir=args.consumed_pairs_dir,
        detailed_timing_recorder=detailed_timing if detailed_timing.enabled else None,
        callbacks=callbacks or None,
    )
    detailed_timing.event("trainer_constructed")
    if (
        args.resume_from_checkpoint
        and args.preference_objective == "dpo"
        and args.chosen_nll_scale_mode == "dpo_saturation"
    ):
        state_path = Path(args.resume_from_checkpoint) / "dpo_trainer_state.json"
        if state_path.is_file():
            trainer._nll_scale = float(json.loads(state_path.read_text(encoding="utf-8"))["nll_scale"])
            print(f"[resume] restored nll_scale={trainer._nll_scale:.6g} from {state_path}", flush=True)
        else:
            # Resuming with a fresh scale would restart the regularizer at full
            # strength on an already-saturated policy, which is exactly the
            # failure this schedule exists to prevent.
            raise RuntimeError(
                f"Resuming a dpo_saturation run but {state_path} is missing. The NLL scale would reset to "
                f"{trainer._nll_scale}, silently re-applying full-strength regularization. Either restart "
                "from step zero or restore the checkpoint's scale by hand."
            )
    detailed_timing.event("trainer_train_call_begin")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    detailed_timing.event("trainer_train_call_end", global_step=trainer.state.global_step)
    final_dir = Path(args.final_adapter_dir)
    if trainer.is_world_process_zero():
        detailed_timing.event("final_adapter_save_begin")
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        write_json(
            Path(args.output_dir).parent / "manifests" / "training_manifest.json",
            {
                "model_path": args.model_path,
                "model_profile": args.model_profile,
                "train_jsonl": args.data_jsonl,
                "eval_jsonl": args.eval_jsonl,
                "final_adapter_dir": str(final_dir),
                "beta": args.beta,
                "preference_objective": args.preference_objective,
                "kto_desirable_weight": args.kto_desirable_weight if args.preference_objective == "kto" else None,
                "kto_undesirable_weight": args.kto_undesirable_weight if args.preference_objective == "kto" else None,
                "kto_kl_score_batch_size": args.kto_kl_score_batch_size if args.preference_objective == "kto" else None,
                "chosen_nll_coef": args.chosen_nll_coef,
                "chosen_nll_mask": args.chosen_nll_mask if args.chosen_nll_coef > 0 else None,
                "chosen_nll_scale_mode": args.chosen_nll_scale_mode if args.chosen_nll_coef > 0 else None,
                "chosen_nll_scale_ema": args.chosen_nll_scale_ema if args.chosen_nll_coef > 0 else None,
                "length_normalize_dpo": bool(args.length_normalize_dpo),
                **chat_rendering_metadata(args),
                **td_dpo_reference_metadata(args),
                "dpop_penalty_weight": args.dpop_penalty_weight,
                "ddp_timeout_seconds": args.ddp_timeout_seconds,
                "synchronize_checkpoints": args.synchronize_checkpoints,
                "checkpoint_complete_marker": CHECKPOINT_COMPLETE_MARKER,
                "adapter_snapshot_steps": snapshot_steps,
                "adapter_snapshot_dir": str(snapshot_dir) if snapshot_steps else None,
                "consumed_pairs_dir": args.consumed_pairs_dir,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "num_train_epochs": args.num_train_epochs,
                "microbatch_size": args.microbatch_size,
                "grad_accum": args.grad_accum,
                "max_length": args.max_length,
                "assistant_loss_mask": args.assistant_loss_mask,
                "seed": args.seed,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "step_timing_dir": args.step_timing_dir,
                "detailed_timing_dir": args.detailed_timing_dir,
                "git_revision": git_revision(),
            },
        )
        print(f"[done] final_adapter={final_dir}")
        detailed_timing.event("final_adapter_save_end")
    detailed_timing.event("train_function_end", global_step=trainer.state.global_step)
    return 0


def main() -> int:
    args = resolve_profile(parse_args())
    if args.preference_objective == "kto" and args.mode != "train":
        raise ValueError(
            "KTO calculates its frozen-reference and cyclic mismatch scores online during train mode; "
            "do not run it through precompute-reference or attach-reference."
        )
    if args.mode == "precompute-reference":
        return precompute_reference(args)
    if args.mode == "attach-reference":
        return attach_reference(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
