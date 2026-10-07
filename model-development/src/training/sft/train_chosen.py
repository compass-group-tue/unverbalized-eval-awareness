#!/usr/bin/env python3
"""Trajectory-aware chosen-only SFT for the special-token DPO data.

This is deliberately separate from the historical mixed SDF/SFT trainer.  It
accepts the chosen trajectories emitted by ``build_special_token_sft_dpo_split``
and uses the same selectable Qwen chat rendering as the special-token DPO
trainer.
The loss-mask setting controls whether SFT supervises only real assistant text
or the complete rendered assistant turn (including its closing delimiters and
turn-termination tokens).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from training.common.model_profiles import MODEL_PROFILES
from training.dpo.train import (
    ASSISTANT_LOSS_MASK_CHOICES,
    CHAT_RENDERING_CHOICES,
    CHECKPOINT_COMPLETE_MARKER,
    DEFAULT_CHAT_RENDERING,
    LABEL_PAD_TOKEN_ID,
    build_model,
    chat_rendering_metadata,
    distributed_barrier,
    render_sequence,
    selected_token_logps,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--output-dir", required=True, help="Checkpoint directory.")
    parser.add_argument("--final-adapter-dir", required=True)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="qwen3_5_27b")
    parser.add_argument("--model-path")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--target-modules", nargs="+")
    parser.add_argument("--max-length", type=int, default=10240)
    parser.add_argument(
        "--assistant-loss-mask",
        choices=ASSISTANT_LOSS_MASK_CHOICES,
        default="all_assistant",
        help=(
            "all_assistant (default; the paper model) supervises the full rendered trainable assistant "
            "turn, including native formatting/termination tokens; content_only supervises just "
            "reasoning_content and visible content."
        ),
    )
    parser.add_argument(
        "--chat-rendering",
        choices=CHAT_RENDERING_CHOICES,
        default=DEFAULT_CHAT_RENDERING,
        help=(
            "trl_qwen3_5_training (default; the paper model) uses TRL 1.9.0's Qwen3.5 training "
            "template and direct final-completion mask. legacy_prefix_delta reproduces pre-V13 SFT runs."
        ),
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, default=1)
    # Paper model: global batch four (four GPUs x microbatch 1 x accumulation 1)
    # at learning rate 1e-5.
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=3600)
    parser.add_argument("--synchronize-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def has_empty_trainable_message(row: dict[str, Any]) -> bool:
    for segment in row.get("chosen_segments") or []:
        if not isinstance(segment, list):
            continue
        for message in segment:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant" or message.get("_dpo_trainable") is not True:
                continue
            if str(message.get("reasoning_content") or "") or str(message.get("content") or ""):
                continue
            return True
    return False


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    source_ids: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_pair_id") or row.get("sft_id") or "")
        if not source_id:
            raise ValueError(f"{path}: SFT row is missing source_pair_id/sft_id")
        if source_id in source_ids:
            raise ValueError(f"{path}: duplicate SFT source_pair_id {source_id!r}")
        source_ids.add(source_id)
        if not isinstance(row.get("chosen_segments"), list) or not row["chosen_segments"]:
            raise ValueError(f"{path}: {source_id}: chosen_segments is missing or empty")
        if has_empty_trainable_message(row):
            raise ValueError(f"{path}: {source_id}: empty trainable chosen assistant message")
    return rows


class ChosenTrajectoryDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        assistant_loss_mask: str,
        chat_rendering: str,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_loss_mask = assistant_loss_mask
        self.chat_rendering = chat_rendering

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source_pair_id = str(row.get("source_pair_id") or row.get("sft_id"))
        segments: list[dict[str, torch.Tensor]] = []
        for segment_index, messages in enumerate(row["chosen_segments"]):
            if not isinstance(messages, list):
                raise RuntimeError(f"{source_pair_id}: chosen_segments[{segment_index}] is not a message list")
            input_ids, labels = render_sequence(
                self.tokenizer,
                messages,
                assistant_loss_mask=self.assistant_loss_mask,
                chat_rendering=self.chat_rendering,
            )
            if len(input_ids) > self.max_length:
                raise RuntimeError(
                    f"{source_pair_id}: chosen_segments[{segment_index}] exceeds --max-length {self.max_length}"
                )
            segments.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        return {"source_pair_id": source_pair_id, "segments": segments}


class ChosenTrajectoryCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    @staticmethod
    def _pad(items: list[torch.Tensor], value: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(item.shape[0]) for item in items)
        values: list[torch.Tensor] = []
        attention: list[torch.Tensor] = []
        for item in items:
            padding = max_len - int(item.shape[0])
            if padding:
                item = torch.cat([item, torch.full((padding,), value, dtype=torch.long)])
            values.append(item)
            attention.append(
                torch.cat(
                    [torch.ones(max_len - padding, dtype=torch.long), torch.zeros(padding, dtype=torch.long)]
                )
            )
        return torch.stack(values), torch.stack(attention)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        owners: list[int] = []
        for owner, feature in enumerate(features):
            for segment in feature["segments"]:
                input_ids.append(segment["input_ids"])
                labels.append(segment["labels"])
                owners.append(owner)
        padded_ids, attention_mask = self._pad(input_ids, self.pad_token_id)
        padded_labels, _ = self._pad(labels, LABEL_PAD_TOKEN_ID)
        return {
            "source_pair_id": [feature["source_pair_id"] for feature in features],
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "segment_owner": torch.tensor(owners, dtype=torch.long),
        }


class ChosenTrajectorySFTTrainer(Trainer):
    """Token-mean NLL over all trainable chosen trajectory segments."""

    def __init__(self, *args, synchronize_checkpoints: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.synchronize_checkpoints = synchronize_checkpoints
        self._last_metrics: dict[str, float] = {}

    def _save_checkpoint(self, model, trial, *args, **kwargs):  # type: ignore[override]
        step = int(self.state.global_step)
        result = super()._save_checkpoint(model, trial, *args, **kwargs)
        if self.synchronize_checkpoints:
            distributed_barrier()
        if self.is_world_process_zero():
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{step}"
            if checkpoint_dir.is_dir():
                (checkpoint_dir / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")
                print(f"[checkpoint] complete step={step} dir={checkpoint_dir}", flush=True)
        if self.synchronize_checkpoints:
            distributed_barrier()
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        token_logps, valid = selected_token_logps(outputs.logits, inputs["labels"])
        token_count = valid.sum()
        if int(token_count.item()) <= 0:
            raise RuntimeError("Chosen SFT batch has no content-only trainable tokens")
        loss = -(token_logps * valid).sum() / token_count
        with torch.no_grad():
            self._last_metrics = {
                "sft/loss": float(loss.detach().cpu()),
                "sft/trainable_tokens": float(token_count.detach().cpu()),
                "objective/total_loss": float(loss.detach().cpu()),
            }
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):  # type: ignore[override]
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:  # type: ignore[override]
        logs = dict(logs)
        if not any(key.startswith("eval_") for key in logs):
            logs.update(self._last_metrics)
        return super().log(logs, start_time=start_time)


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    init_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": False,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.microbatch_size,
        "per_device_eval_batch_size": 1,
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
        "eval_steps": args.eval_steps,
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


def train(args: argparse.Namespace) -> int:
    if args.max_length <= 0 or args.microbatch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("max length, microbatch size, and grad accumulation must be positive")
    train_rows = load_rows(Path(args.data_jsonl))
    eval_rows = load_rows(Path(args.eval_jsonl)) if args.eval_jsonl else None
    train_ids = {str(row.get("source_pair_id") or row.get("sft_id")) for row in train_rows}
    eval_ids = {str(row.get("source_pair_id") or row.get("sft_id")) for row in (eval_rows or [])}
    if train_ids & eval_ids:
        raise ValueError("SFT train/eval source_pair_id overlap")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ChosenTrajectoryDataset(
        train_rows, tokenizer, args.max_length, args.assistant_loss_mask, args.chat_rendering
    )
    eval_dataset = (
        ChosenTrajectoryDataset(
            eval_rows, tokenizer, args.max_length, args.assistant_loss_mask, args.chat_rendering
        )
        if eval_rows
        else None
    )
    collator = ChosenTrajectoryCollator(tokenizer.pad_token_id)

    model = build_model(args, with_lora=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Chosen-trajectory SFT requires a CUDA GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    model.to(torch.device("cuda", local_rank))
    trainer = ChosenTrajectorySFTTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        synchronize_checkpoints=args.synchronize_checkpoints,
        callbacks=None,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = Path(args.final_adapter_dir)
    if trainer.is_world_process_zero():
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
                "objective": f"chosen_trajectory_{args.assistant_loss_mask}_sft",
                "assistant_loss_mask": args.assistant_loss_mask,
                **chat_rendering_metadata(args),
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "num_train_epochs": args.num_train_epochs,
                "microbatch_size": args.microbatch_size,
                "grad_accum": args.grad_accum,
                "max_length": args.max_length,
                "seed": args.seed,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "synchronize_checkpoints": args.synchronize_checkpoints,
                "checkpoint_complete_marker": CHECKPOINT_COMPLETE_MARKER,
                "git_revision": git_revision(),
            },
        )
        print(f"[done] final_adapter={final_dir}", flush=True)
    return 0


def main() -> int:
    return train(resolve_profile(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
