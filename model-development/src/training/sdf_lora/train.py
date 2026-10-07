#!/usr/bin/env python3
"""
LoRA training for mixed SDF + SFT data with:
- on-the-fly tokenization (no prebuilt token packs in RAM)
- one-doc-per-window samples
- two length buckets (<4k, >=4k truncated to 16k)
- random SDF/SFT interleaving with window cap (max 8 of one type in 10)
- HP mode (loss-token-budget stop) and Full mode (checkpointing + optional val continuation)
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from training.common.model_profiles import MODEL_PROFILES

os.environ.setdefault("SOFT_FILELOCK", "1")


@dataclass(frozen=True)
class Record:
    path: str
    offset: int
    source_type: str  # sdf|sft
    source_dataset: str
    objective: str
    is_reasoning_sft: bool
    est_tokens: int
    len_band: str  # low|mid|long


@dataclass(frozen=True)
class SamplePlan:
    rec: Record
    bucket: str  # short|long
    max_seq_len: int


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    mm_token_type_ids: list[int]
    source_type: str
    source_dataset: str
    trainable_tokens: int
    bucket: str
    max_seq_len: int


@dataclass(frozen=True)
class SFTThinkingConfig:
    control_placement: str  # off|system|user
    reasoning_control: str
    non_reasoning_control: str
    pass_enable_thinking: bool


DEFAULT_SFT_THINKING_CONFIG = SFTThinkingConfig(
    control_placement="system",
    reasoning_control="/think",
    non_reasoning_control="/no_think",
    pass_enable_thinking=True,
)


_ENABLE_THINKING_FALLBACK_WARNED = False


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def is_reasoning_source(source_dataset: str) -> bool:
    return source_dataset.startswith("am_qwen3_distilled_") or source_dataset.startswith("smoltalk_")


def iter_jsonl_with_offsets(path: Path):
    with path.open("rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                yield off, row, None
            except Exception as e:  # noqa: BLE001
                yield off, None, str(e)


def estimate_tokens_from_row(row: dict[str, Any], source_type: str) -> int:
    # Lightweight estimate used for bucket planning; true tokenization happens on-the-fly at training.
    if source_type == "sdf":
        text = row.get("text")
        chars = len(text) if isinstance(text, str) else 0
    else:
        msgs = row.get("messages")
        chars = 0
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                c = m.get("content", "")
                if isinstance(c, str):
                    chars += len(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, str):
                            chars += len(part)
                        elif isinstance(part, dict):
                            t = part.get("text")
                            if isinstance(t, str):
                                chars += len(t)
    # Rough heuristic.
    return max(1, chars // 4)


def band_for_est_tokens(n: int) -> str:
    if n < 1000:
        return "low"
    if n < 4000:
        return "mid"
    return "long"


def build_records(data_root: Path, max_records_per_split: int | None = None) -> tuple[list[Record], list[Record], dict[str, Any]]:
    sdf_records: list[Record] = []
    sft_records: list[Record] = []
    parse_errors = 0
    skipped_empty = 0

    sdf_files = sorted((data_root / "sdf").glob("*.jsonl"))
    sft_files = sorted((data_root / "sft").glob("*.jsonl"))
    if not sdf_files and not sft_files:
        raise RuntimeError(f"No dataset files found under {data_root}/sdf or {data_root}/sft")

    for p in sdf_files:
        n = 0
        for off, row, err in iter_jsonl_with_offsets(p):
            if err is not None:
                parse_errors += 1
                continue
            if not isinstance(row, dict):
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                skipped_empty += 1
                continue
            src = str(row.get("source_dataset", p.stem))
            est = estimate_tokens_from_row(row, "sdf")
            sdf_records.append(
                Record(
                    path=str(p),
                    offset=off,
                    source_type="sdf",
                    source_dataset=src,
                    objective=str(row.get("objective", "sdf_full_loss")),
                    is_reasoning_sft=False,
                    est_tokens=est,
                    len_band=band_for_est_tokens(est),
                )
            )
            n += 1
            if max_records_per_split is not None and n >= max_records_per_split:
                break

    for p in sft_files:
        n = 0
        for off, row, err in iter_jsonl_with_offsets(p):
            if err is not None:
                parse_errors += 1
                continue
            if not isinstance(row, dict):
                continue
            msgs = row.get("messages")
            if not isinstance(msgs, list) or not msgs:
                skipped_empty += 1
                continue
            src = str(row.get("source_dataset", p.stem))
            est = estimate_tokens_from_row(row, "sft")
            sft_records.append(
                Record(
                    path=str(p),
                    offset=off,
                    source_type="sft",
                    source_dataset=src,
                    objective=str(row.get("objective", "sft_assistant_only")),
                    is_reasoning_sft=is_reasoning_source(src),
                    est_tokens=est,
                    len_band=band_for_est_tokens(est),
                )
            )
            n += 1
            if max_records_per_split is not None and n >= max_records_per_split:
                break

    summary = {
        "sdf_records": len(sdf_records),
        "sft_records": len(sft_records),
        "parse_errors": parse_errors,
        "skipped_empty_rows": skipped_empty,
        "sdf_files": [str(x) for x in sdf_files],
        "sft_files": [str(x) for x in sft_files],
    }
    return sdf_records, sft_records, summary


def choose_next_type(
    window: deque[str],
    rem_sdf: int,
    rem_sft: int,
    rng: random.Random,
    max_same_in_window: int,
    window_size: int,
    enforce_window_cap_minority_floor: float,
) -> str:
    if rem_sdf <= 0 and rem_sft <= 0:
        raise RuntimeError("No records remaining.")
    if rem_sdf <= 0:
        return "sft"
    if rem_sft <= 0:
        return "sdf"

    total = rem_sdf + rem_sft
    minority_share = min(rem_sdf, rem_sft) / total
    if len(window) >= window_size and minority_share >= enforce_window_cap_minority_floor:
        sdf_count = sum(1 for x in window if x == "sdf")
        sft_count = len(window) - sdf_count
        if sdf_count >= max_same_in_window:
            return "sft"
        if sft_count >= max_same_in_window:
            return "sdf"

    p_sdf = rem_sdf / total
    return "sdf" if rng.random() < p_sdf else "sft"


def split_pools(records: list[Record], rng: random.Random) -> dict[str, list[Record]]:
    low = [r for r in records if r.len_band == "low"]
    mid = [r for r in records if r.len_band == "mid"]
    long = [r for r in records if r.len_band == "long"]
    rng.shuffle(low)
    rng.shuffle(mid)
    rng.shuffle(long)
    return {"low": low, "mid": mid, "long": long}


def count_pool_items(pools: dict[str, list[Record]]) -> int:
    return len(pools["low"]) + len(pools["mid"]) + len(pools["long"])


def pop_for_bucket(
    pools: dict[str, list[Record]],
    bucket: str,
    long_mid_used: int,
    long_total_used: int,
    long_mid_bleed_ratio: float,
) -> tuple[Record | None, str, int, int]:
    # Returns (record, final_bucket, long_mid_used, long_total_used)
    low, mid, long = pools["low"], pools["mid"], pools["long"]
    if bucket == "long":
        long_total_used += 1
        use_mid = (long_mid_used / max(1, long_total_used)) < long_mid_bleed_ratio and len(mid) > 0
        if use_mid:
            long_mid_used += 1
            return mid.pop(), "long", long_mid_used, long_total_used
        if long:
            return long.pop(), "long", long_mid_used, long_total_used
        if mid:
            long_mid_used += 1
            return mid.pop(), "long", long_mid_used, long_total_used
        if low:
            # Fallback if nothing else remains.
            return low.pop(), "short", long_mid_used, long_total_used
        return None, "long", long_mid_used, long_total_used

    # short bucket
    if low:
        return low.pop(), "short", long_mid_used, long_total_used
    if mid:
        return mid.pop(), "short", long_mid_used, long_total_used
    if long:
        return long.pop(), "long", long_mid_used, long_total_used
    return None, "short", long_mid_used, long_total_used


def build_sample_plan(
    *,
    sdf_records: list[Record],
    sft_records: list[Record],
    seed: int,
    max_same_in_window: int,
    window_size: int,
    enforce_window_cap_minority_floor: float,
    long_mid_bleed_ratio: float,
    short_seq_len: int,
    long_seq_len: int,
) -> tuple[list[SamplePlan], dict[str, Any]]:
    rng = random.Random(seed)
    sdf_pools = split_pools(sdf_records, rng)
    sft_pools = split_pools(sft_records, rng)

    window: deque[str] = deque(maxlen=window_size)
    plans: list[SamplePlan] = []
    used_by_type = Counter()
    used_by_bucket = Counter()

    long_mid_used = {"sdf": 0, "sft": 0}
    long_total_used = {"sdf": 0, "sft": 0}

    def rem_count(which: str) -> int:
        return count_pool_items(sdf_pools) if which == "sdf" else count_pool_items(sft_pools)

    while rem_count("sdf") > 0 or rem_count("sft") > 0:
        chosen_type = choose_next_type(
            window=window,
            rem_sdf=rem_count("sdf"),
            rem_sft=rem_count("sft"),
            rng=rng,
            max_same_in_window=max_same_in_window,
            window_size=window_size,
            enforce_window_cap_minority_floor=enforce_window_cap_minority_floor,
        )
        pools = sdf_pools if chosen_type == "sdf" else sft_pools
        short_avail = len(pools["low"]) + len(pools["mid"])
        long_avail = len(pools["long"]) + len(pools["mid"])
        if short_avail <= 0 and long_avail <= 0:
            continue

        if short_avail > 0 and long_avail > 0:
            # Favor long when true-long items remain.
            p_long = len(pools["long"]) / max(1, len(pools["long"]) + short_avail)
            bucket = "long" if rng.random() < p_long else "short"
        elif long_avail > 0:
            bucket = "long"
        else:
            bucket = "short"

        rec, final_bucket, long_mid_used[chosen_type], long_total_used[chosen_type] = pop_for_bucket(
            pools=pools,
            bucket=bucket,
            long_mid_used=long_mid_used[chosen_type],
            long_total_used=long_total_used[chosen_type],
            long_mid_bleed_ratio=long_mid_bleed_ratio,
        )
        if rec is None:
            continue

        max_seq_len = long_seq_len if final_bucket == "long" else short_seq_len
        plans.append(SamplePlan(rec=rec, bucket=final_bucket, max_seq_len=max_seq_len))
        used_by_type[chosen_type] += 1
        used_by_bucket[final_bucket] += 1
        window.append(chosen_type)

        if len(plans) % 10000 == 0:
            rank0_print(
                f"[plan] built={len(plans)} rem_sdf={rem_count('sdf')} rem_sft={rem_count('sft')} "
                f"long_ratio_so_far={used_by_bucket['long']/max(1,len(plans)):.3f}"
            )

    summary = {
        "num_samples": len(plans),
        "used_by_type": dict(used_by_type),
        "used_by_bucket": dict(used_by_bucket),
        "long_mid_used": long_mid_used,
        "long_total_used": long_total_used,
    }
    return plans, summary


def load_row_from_offset(handles: dict[str, Any], rec: Record) -> dict[str, Any] | None:
    h = handles.get(rec.path)
    if h is None:
        h = open(rec.path, "rb")
        handles[rec.path] = h
    h.seek(rec.offset)
    line = h.readline()
    if not line:
        return None
    try:
        row = json.loads(line)
    except Exception:
        return None
    return row if isinstance(row, dict) else None


def ensure_list_1d(x: Any) -> list[int]:
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [int(t) for t in x[0]]
    if isinstance(x, list):
        return [int(t) for t in x]
    raise TypeError(f"Cannot convert to 1D list: {type(x)}")


def _control_line_present(text: str, control: str) -> bool:
    c = (control or "").strip()
    if not c:
        return True
    return any(line.strip() == c for line in text.splitlines())


def _append_control_to_text(text: str, control: str) -> str:
    c = (control or "").strip()
    if not c:
        return text
    if _control_line_present(text, c):
        return text
    t = text.strip()
    if not t:
        return c
    return f"{t}\n{c}"


def apply_sft_thinking_controls(
    messages: list[dict[str, Any]],
    *,
    is_reasoning_sft: bool,
    cfg: SFTThinkingConfig,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = [dict(m) for m in messages if isinstance(m, dict)]
    if cfg.control_placement == "off":
        return copied

    control = cfg.reasoning_control if is_reasoning_sft else cfg.non_reasoning_control
    control = (control or "").strip()
    if not control:
        return copied

    if cfg.control_placement == "user":
        for i in range(len(copied) - 1, -1, -1):
            if copied[i].get("role") != "user":
                continue
            content = copied[i].get("content")
            if isinstance(content, str):
                copied[i]["content"] = _append_control_to_text(content, control)
                return copied
        # Fallback if user message is missing/non-text.
        return [{"role": "system", "content": control}, *copied]

    # system placement (default)
    if copied and copied[0].get("role") == "system" and isinstance(copied[0].get("content"), str):
        copied[0]["content"] = _append_control_to_text(copied[0]["content"], control)
        return copied
    return [{"role": "system", "content": control}, *copied]


def render_chat_input_ids(tokenizer, messages: list[dict[str, Any]], *, enable_thinking: bool | None) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if enable_thinking is None:
            raise
        global _ENABLE_THINKING_FALLBACK_WARNED
        if not _ENABLE_THINKING_FALLBACK_WARNED:
            rank0_print("[warn] tokenizer.apply_chat_template does not support enable_thinking; continuing without it.")
            _ENABLE_THINKING_FALLBACK_WARNED = True
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if "input_ids" not in rendered:
        raise RuntimeError("Chat template tokenization did not return input_ids.")
    return ensure_list_1d(rendered["input_ids"])


def render_chat_inputs(tokenizer, messages: list[dict[str, Any]], *, enable_thinking: bool | None) -> dict[str, list[int]]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if enable_thinking is None:
            raise
        global _ENABLE_THINKING_FALLBACK_WARNED
        if not _ENABLE_THINKING_FALLBACK_WARNED:
            rank0_print("[warn] tokenizer.apply_chat_template does not support enable_thinking; continuing without it.")
            _ENABLE_THINKING_FALLBACK_WARNED = True
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if "input_ids" not in rendered:
        raise RuntimeError("Chat template tokenization did not return input_ids.")
    out = {"input_ids": ensure_list_1d(rendered["input_ids"])}
    mm_token_type_ids = rendered.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        out["mm_token_type_ids"] = [0] * len(out["input_ids"])
    else:
        out["mm_token_type_ids"] = ensure_list_1d(mm_token_type_ids)
    return out


def build_assistant_mask_from_prefix_deltas(
    tokenizer,
    messages: list[dict[str, Any]],
    full_len: int,
    *,
    enable_thinking: bool | None,
) -> list[int]:
    mask = [0] * full_len
    prev_ids: list[int] = []
    for i, msg in enumerate(messages):
        cur_ids = render_chat_input_ids(tokenizer, messages[: i + 1], enable_thinking=enable_thinking)
        start = len(prev_ids)
        end = min(len(cur_ids), full_len)
        if msg.get("role") == "assistant" and end > start:
            for j in range(start, end):
                mask[j] = 1
        prev_ids = cur_ids
    return mask


def truncate_pair(
    input_ids: list[int],
    labels: list[int],
    mm_token_type_ids: list[int],
    max_seq_len: int,
    mode: str,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    if len(input_ids) <= max_seq_len:
        return input_ids, labels, mm_token_type_ids
    if mode == "keep_end":
        start = len(input_ids) - max_seq_len
    elif mode == "random_window":
        start = rng.randint(0, len(input_ids) - max_seq_len)
    else:
        raise ValueError(f"Unknown truncation mode: {mode}")
    end = start + max_seq_len
    return input_ids[start:end], labels[start:end], mm_token_type_ids[start:end]


def count_trainable_labels(labels: list[int]) -> int:
    return sum(1 for x in labels if x != -100)


def has_min_train_seq_len(input_ids: list[int], min_len: int = 2) -> bool:
    return len(input_ids) >= min_len


def tokenize_sdf_row(
    tokenizer,
    row: dict[str, Any],
    max_seq_len: int,
    rng: random.Random,
    bucket: str,
    *,
    sdf_doc_prefix: str,
    mask_sdf_doc_prefix: bool,
) -> TokenizedExample | None:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    input_ids = ensure_list_1d(tokenizer.encode(text, add_special_tokens=False))
    labels = input_ids.copy()
    if sdf_doc_prefix:
        prefix_ids = ensure_list_1d(tokenizer.encode(sdf_doc_prefix, add_special_tokens=False))
        input_ids = prefix_ids + input_ids
        if mask_sdf_doc_prefix:
            labels = ([-100] * len(prefix_ids)) + labels
        else:
            labels = prefix_ids + labels
    mm_token_type_ids = [0] * len(input_ids)
    input_ids, labels, mm_token_type_ids = truncate_pair(
        input_ids,
        labels,
        mm_token_type_ids,
        max_seq_len,
        "random_window",
        rng,
    )
    if not has_min_train_seq_len(input_ids):
        return None
    return TokenizedExample(
        input_ids=input_ids,
        labels=labels,
        mm_token_type_ids=mm_token_type_ids,
        source_type="sdf",
        source_dataset=str(row.get("source_dataset", "unknown")),
        trainable_tokens=count_trainable_labels(labels),
        bucket=bucket,
        max_seq_len=max_seq_len,
    )


def tokenize_sft_row(
    tokenizer,
    row: dict[str, Any],
    rec: Record,
    max_seq_len: int,
    rng: random.Random,
    bucket: str,
    sft_thinking_cfg: SFTThinkingConfig | None = None,
) -> TokenizedExample | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    cfg = sft_thinking_cfg or DEFAULT_SFT_THINKING_CONFIG
    controlled_messages = apply_sft_thinking_controls(messages, is_reasoning_sft=rec.is_reasoning_sft, cfg=cfg)
    enable_thinking = bool(rec.is_reasoning_sft) if cfg.pass_enable_thinking else None
    rendered = render_chat_inputs(tokenizer, controlled_messages, enable_thinking=enable_thinking)
    input_ids = rendered["input_ids"]
    mm_token_type_ids = rendered["mm_token_type_ids"]
    assistant_mask = build_assistant_mask_from_prefix_deltas(
        tokenizer,
        controlled_messages,
        len(input_ids),
        enable_thinking=enable_thinking,
    )
    labels = [tid if int(m) == 1 else -100 for tid, m in zip(input_ids, assistant_mask)]
    # SFT policy: keep end when truncating.
    input_ids, labels, mm_token_type_ids = truncate_pair(
        input_ids,
        labels,
        mm_token_type_ids,
        max_seq_len,
        "keep_end",
        rng,
    )
    if not has_min_train_seq_len(input_ids):
        return None
    trainable = count_trainable_labels(labels)
    if trainable <= 0:
        return None
    return TokenizedExample(
        input_ids=input_ids,
        labels=labels,
        mm_token_type_ids=mm_token_type_ids,
        source_type="sft",
        source_dataset=str(row.get("source_dataset", rec.source_dataset)),
        trainable_tokens=trainable,
        bucket=bucket,
        max_seq_len=max_seq_len,
    )


class PlannedTrainingDataset(Dataset):
    def __init__(
        self,
        plans: list[SamplePlan],
        tokenizer,
        seed: int,
        sft_thinking_cfg: SFTThinkingConfig,
        *,
        sdf_doc_prefix: str,
        mask_sdf_doc_prefix: bool,
    ):
        self.plans = plans
        self.tokenizer = tokenizer
        self.seed = seed
        self.sft_thinking_cfg = sft_thinking_cfg
        self.sdf_doc_prefix = sdf_doc_prefix
        self.mask_sdf_doc_prefix = mask_sdf_doc_prefix
        self.handles: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.plans)

    def __del__(self):
        for h in self.handles.values():
            try:
                h.close()
            except Exception:
                pass

    def __getitem__(self, idx: int) -> dict[str, Any]:
        plan = self.plans[idx]
        row = load_row_from_offset(self.handles, plan.rec)
        if row is None:
            # Extremely rare fallback to all-pad masked sample.
            sft_reasoning_flag = -1
            if plan.rec.source_type == "sft":
                sft_reasoning_flag = 1 if plan.rec.is_reasoning_sft else 0
            return {
                "input_ids": torch.tensor([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id], dtype=torch.long),
                "labels": torch.tensor([-100, -100], dtype=torch.long),
                "mm_token_type_ids": torch.tensor([0, 0], dtype=torch.long),
                "pack_type_id": torch.tensor(0 if plan.rec.source_type == "sdf" else 1, dtype=torch.long),
                "nonpad_tokens": torch.tensor(1, dtype=torch.long),
                "trainable_tokens": torch.tensor(0, dtype=torch.long),
                "sft_reasoning_flag": torch.tensor(sft_reasoning_flag, dtype=torch.long),
                "bucket_id": torch.tensor(0 if plan.bucket == "short" else 1, dtype=torch.long),
                "max_seq_len": torch.tensor(plan.max_seq_len, dtype=torch.long),
            }
        rng = random.Random(self.seed + idx)
        if plan.rec.source_type == "sdf":
            ex = tokenize_sdf_row(
                self.tokenizer,
                row,
                plan.max_seq_len,
                rng,
                plan.bucket,
                sdf_doc_prefix=self.sdf_doc_prefix,
                mask_sdf_doc_prefix=self.mask_sdf_doc_prefix,
            )
        else:
            ex = tokenize_sft_row(
                self.tokenizer,
                row,
                plan.rec,
                plan.max_seq_len,
                rng,
                plan.bucket,
                sft_thinking_cfg=self.sft_thinking_cfg,
            )

        if ex is None:
            sft_reasoning_flag = -1
            if plan.rec.source_type == "sft":
                sft_reasoning_flag = 1 if plan.rec.is_reasoning_sft else 0
            return {
                "input_ids": torch.tensor([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id], dtype=torch.long),
                "labels": torch.tensor([-100, -100], dtype=torch.long),
                "mm_token_type_ids": torch.tensor([0, 0], dtype=torch.long),
                "pack_type_id": torch.tensor(0 if plan.rec.source_type == "sdf" else 1, dtype=torch.long),
                "nonpad_tokens": torch.tensor(1, dtype=torch.long),
                "trainable_tokens": torch.tensor(0, dtype=torch.long),
                "sft_reasoning_flag": torch.tensor(sft_reasoning_flag, dtype=torch.long),
                "bucket_id": torch.tensor(0 if plan.bucket == "short" else 1, dtype=torch.long),
                "max_seq_len": torch.tensor(plan.max_seq_len, dtype=torch.long),
            }

        sft_reasoning_flag = -1
        if plan.rec.source_type == "sft":
            sft_reasoning_flag = 1 if plan.rec.is_reasoning_sft else 0
        return {
            "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
            "labels": torch.tensor(ex.labels, dtype=torch.long),
            "mm_token_type_ids": torch.tensor(ex.mm_token_type_ids, dtype=torch.long),
            "pack_type_id": torch.tensor(0 if ex.source_type == "sdf" else 1, dtype=torch.long),
            "nonpad_tokens": torch.tensor(len(ex.input_ids), dtype=torch.long),
            "trainable_tokens": torch.tensor(ex.trainable_tokens, dtype=torch.long),
            "sft_reasoning_flag": torch.tensor(sft_reasoning_flag, dtype=torch.long),
            "bucket_id": torch.tensor(0 if ex.bucket == "short" else 1, dtype=torch.long),
            "max_seq_len": torch.tensor(ex.max_seq_len, dtype=torch.long),
        }


class DynamicPadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise RuntimeError("Empty batch.")

        bucket_ids = {int(f["bucket_id"].item()) for f in features}
        if len(bucket_ids) > 1:
            raise RuntimeError(f"Mixed bucket batch detected: {bucket_ids}")

        max_len = max(int(f["input_ids"].shape[0]) for f in features)
        out_ids = []
        out_labels = []
        out_mm_token_type_ids = []
        out_mask = []
        pack_type = []
        nonpad = []
        trainable = []
        sft_reasoning = []
        bucket = []
        max_seq_lens = []
        for f in features:
            ids = f["input_ids"]
            labels = f["labels"]
            mm_token_type_ids = f["mm_token_type_ids"]
            cur = int(ids.shape[0])
            pad = max_len - cur
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=torch.long)], dim=0)
                labels = torch.cat([labels, torch.full((pad,), -100, dtype=torch.long)], dim=0)
                mm_token_type_ids = torch.cat([mm_token_type_ids, torch.zeros((pad,), dtype=torch.long)], dim=0)
            mask = torch.cat([torch.ones((cur,), dtype=torch.long), torch.zeros((pad,), dtype=torch.long)], dim=0)
            out_ids.append(ids)
            out_labels.append(labels)
            out_mm_token_type_ids.append(mm_token_type_ids)
            out_mask.append(mask)
            pack_type.append(f["pack_type_id"])
            nonpad.append(f["nonpad_tokens"])
            trainable.append(f["trainable_tokens"])
            sft_reasoning.append(f["sft_reasoning_flag"])
            bucket.append(f["bucket_id"])
            max_seq_lens.append(f["max_seq_len"])

        return {
            "input_ids": torch.stack(out_ids, dim=0),
            "labels": torch.stack(out_labels, dim=0),
            "mm_token_type_ids": torch.stack(out_mm_token_type_ids, dim=0),
            "attention_mask": torch.stack(out_mask, dim=0),
            "pack_type_id": torch.stack(pack_type, dim=0),
            "nonpad_tokens": torch.stack(nonpad, dim=0),
            "trainable_tokens": torch.stack(trainable, dim=0),
            "sft_reasoning_flag": torch.stack(sft_reasoning, dim=0),
            "bucket_id": torch.stack(bucket, dim=0),
            "max_seq_len": torch.stack(max_seq_lens, dim=0),
        }


def p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = int(math.ceil(0.95 * len(s))) - 1
    k = max(0, min(k, len(s) - 1))
    return float(s[k])


class BlockAttentionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = time.time()
        self.total_nonpad_tokens_seen = 0
        self.total_trainable_tokens_seen = 0
        self.total_trainable_tokens_sdf = 0
        self.total_trainable_tokens_sft = 0
        self.total_trainable_tokens_reasoning_sft = 0
        self.total_trainable_tokens_non_reasoning_sft = 0
        self.recent_step_durations = deque(maxlen=200)
        self.recent_seq_lens = deque(maxlen=200)
        self.recent_loss_sdf = deque(maxlen=200)
        self.recent_loss_sft = deque(maxlen=200)
        self.recent_bucket = deque(maxlen=200)
        self.last_batch_type = None
        self.last_nonpad_tokens = 0
        self.last_trainable_tokens = 0
        self.last_trainable_tokens_sdf = 0
        self.last_trainable_tokens_sft = 0
        self.last_trainable_tokens_reasoning_sft = 0
        self.last_trainable_tokens_non_reasoning_sft = 0

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        t0 = time.time()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self.recent_step_durations.append(time.time() - t0)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        pack_type_id = inputs.pop("pack_type_id")
        nonpad_tokens = inputs.pop("nonpad_tokens")
        trainable_tokens = inputs.pop("trainable_tokens")
        sft_reasoning_flag = inputs.pop("sft_reasoning_flag")
        bucket_id = inputs.pop("bucket_id")
        inputs.pop("max_seq_len")

        bsz = int(inputs["input_ids"].shape[0])
        outputs = model(**inputs, use_cache=False)
        loss = outputs.loss

        batch_nonpad = int(nonpad_tokens.sum().item())
        batch_trainable = int(trainable_tokens.sum().item())
        batch_type = int(pack_type_id[0].item())
        loss_value = float(loss.detach().item())
        bucket = int(bucket_id[0].item())
        sdf_mask = pack_type_id == 0
        sft_mask = pack_type_id == 1
        reasoning_sft_mask = sft_mask & (sft_reasoning_flag == 1)
        non_reasoning_sft_mask = sft_mask & (sft_reasoning_flag == 0)
        batch_trainable_sdf = int(trainable_tokens[sdf_mask].sum().item())
        batch_trainable_sft = int(trainable_tokens[sft_mask].sum().item())
        batch_trainable_reasoning_sft = int(trainable_tokens[reasoning_sft_mask].sum().item())
        batch_trainable_non_reasoning_sft = int(trainable_tokens[non_reasoning_sft_mask].sum().item())

        self.total_nonpad_tokens_seen += batch_nonpad
        self.total_trainable_tokens_seen += batch_trainable
        self.total_trainable_tokens_sdf += batch_trainable_sdf
        self.total_trainable_tokens_sft += batch_trainable_sft
        self.total_trainable_tokens_reasoning_sft += batch_trainable_reasoning_sft
        self.total_trainable_tokens_non_reasoning_sft += batch_trainable_non_reasoning_sft
        self.recent_seq_lens.append(batch_nonpad / max(1, bsz))
        self.recent_bucket.append(bucket)
        if batch_type == 0:
            self.recent_loss_sdf.append(loss_value)
            self.last_batch_type = "sdf"
        else:
            self.recent_loss_sft.append(loss_value)
            self.last_batch_type = "sft"
        self.last_nonpad_tokens = batch_nonpad
        self.last_trainable_tokens = batch_trainable
        self.last_trainable_tokens_sdf = batch_trainable_sdf
        self.last_trainable_tokens_sft = batch_trainable_sft
        self.last_trainable_tokens_reasoning_sft = batch_trainable_reasoning_sft
        self.last_trainable_tokens_non_reasoning_sft = batch_trainable_non_reasoning_sft

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None):  # type: ignore[override]
        logs = dict(logs)
        elapsed = time.time() - self.t0
        logs["tokens_seen_nonpad"] = float(self.total_nonpad_tokens_seen)
        logs["tokens_seen_trainable"] = float(self.total_trainable_tokens_seen)
        logs["tokens_seen_trainable_sdf"] = float(self.total_trainable_tokens_sdf)
        logs["tokens_seen_trainable_sft"] = float(self.total_trainable_tokens_sft)
        logs["tokens_seen_trainable_reasoning_sft"] = float(self.total_trainable_tokens_reasoning_sft)
        logs["tokens_seen_trainable_non_reasoning_sft"] = float(self.total_trainable_tokens_non_reasoning_sft)
        logs["tokens_seen_trainable_sdf_frac"] = float(self.total_trainable_tokens_sdf / max(1, self.total_trainable_tokens_seen))
        logs["tokens_seen_trainable_sft_frac"] = float(self.total_trainable_tokens_sft / max(1, self.total_trainable_tokens_seen))
        logs["tokens_seen_trainable_reasoning_sft_frac_of_sft"] = float(
            self.total_trainable_tokens_reasoning_sft / max(1, self.total_trainable_tokens_sft)
        )
        logs["tokens_per_sec_nonpad"] = float(self.total_nonpad_tokens_seen / max(elapsed, 1e-6))
        logs["step_time_mean_s"] = float(sum(self.recent_step_durations) / max(1, len(self.recent_step_durations)))
        logs["step_time_p95_s"] = p95(list(self.recent_step_durations))
        logs["seq_len_mean"] = float(sum(self.recent_seq_lens) / max(1, len(self.recent_seq_lens)))
        logs["seq_len_p90"] = float(sorted(self.recent_seq_lens)[max(0, int(0.9 * len(self.recent_seq_lens)) - 1)]) if self.recent_seq_lens else 0.0
        logs["bucket_long_frac_recent"] = float(sum(self.recent_bucket) / max(1, len(self.recent_bucket)))
        logs["batch_nonpad_tokens"] = float(self.last_nonpad_tokens)
        logs["batch_trainable_tokens"] = float(self.last_trainable_tokens)
        logs["batch_trainable_tokens_sdf"] = float(self.last_trainable_tokens_sdf)
        logs["batch_trainable_tokens_sft"] = float(self.last_trainable_tokens_sft)
        logs["batch_trainable_tokens_reasoning_sft"] = float(self.last_trainable_tokens_reasoning_sft)
        logs["batch_trainable_tokens_non_reasoning_sft"] = float(self.last_trainable_tokens_non_reasoning_sft)
        if self.recent_loss_sdf:
            logs["loss_sdf"] = float(sum(self.recent_loss_sdf) / len(self.recent_loss_sdf))
        if self.recent_loss_sft:
            logs["loss_sft"] = float(sum(self.recent_loss_sft) / len(self.recent_loss_sft))
        if torch.cuda.is_available():
            logs["gpu_mem_peak_gb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        super().log(logs, start_time=start_time)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.args.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = SequentialSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )


class LossTokenBudgetCallback(TrainerCallback):
    def __init__(self, budget: int):
        self.budget = budget
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        if trainer is None:
            return control
        seen_local = int(getattr(trainer, "total_trainable_tokens_seen", 0))
        seen_global = seen_local
        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            t = torch.tensor([seen_local], dtype=torch.long, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            seen_global = int(t.item())
        if seen_global >= self.budget:
            control.should_training_stop = True
            rank0_print(
                f"[hp-stop] GLOBAL loss-token budget reached: global={seen_global} local={seen_local} target={self.budget}"
            )
        return control


class CheckpointQuestionEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        questions_jsonl: str,
        base_model_path: str,
        eval_script_path: str,
        reasoning_system_prompt: str,
        seed: int,
        max_new_tokens: int,
        tensor_parallel_size: int,
    ):
        self.questions_jsonl = questions_jsonl
        self.base_model_path = base_model_path
        self.eval_script_path = eval_script_path
        self.reasoning_system_prompt = reasoning_system_prompt
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size

    def on_save(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", "0")) != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{int(state.global_step)}")
        if not os.path.isdir(ckpt_dir):
            return control
        out_dir = os.path.join(args.output_dir, "eval_questions")
        os.makedirs(out_dir, exist_ok=True)
        out_jsonl = os.path.join(out_dir, f"step_{int(state.global_step)}.jsonl")
        cmd = [
            "python3",
            self.eval_script_path,
            "--questions-jsonl",
            self.questions_jsonl,
            "--out-jsonl",
            out_jsonl,
            "--base-model-path",
            self.base_model_path,
            "--adapter-path",
            ckpt_dir,
            "--seed",
            str(self.seed),
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--reasoning-system-prompt",
            self.reasoning_system_prompt,
            "--control-placement",
            "system",
            "--reasoning-enable-thinking",
            "--no-non-reasoning-enable-thinking",
            "--include-debug-prompts",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        ]
        rank0_print(f"[eval-questions] running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:  # noqa: BLE001
            rank0_print(f"[eval-questions] WARNING failed at step {int(state.global_step)}: {e}")
        return control


def find_latest_checkpoint(output_dir: str) -> str | None:
    p = Path(output_dir)
    if not p.exists():
        return None
    best = None
    best_step = -1
    for c in p.glob("checkpoint-*"):
        if not c.is_dir():
            continue
        try:
            step = int(c.name.split("-")[-1])
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best = str(c)
    return best


def resolve_resume_checkpoint(output_dir: str, explicit: str | None, no_resume: bool) -> str | None:
    if no_resume:
        return None
    if explicit:
        if not os.path.isdir(explicit):
            raise FileNotFoundError(f"--resume-from-checkpoint path does not exist: {explicit}")
        return explicit
    return find_latest_checkpoint(output_dir)


def build_training_args(args, *, for_val_phase: bool = False) -> TrainingArguments:
    params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ddp_find_unused_parameters = args.model_profile == "gemma4_31b"
    save_steps_value = args.save_steps
    if args.run_mode == "hp":
        if not for_val_phase and int(args.hp_save_steps) > 0:
            save_strategy = "steps"
            save_steps_value = int(args.hp_save_steps)
        else:
            save_strategy = "no"
        eval_strategy = "no"
    elif for_val_phase:
        save_strategy = "no"
        eval_strategy = "no"
    else:
        save_strategy = "steps"
        eval_strategy = "no"

    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": False,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.microbatch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": save_strategy,
        "save_steps": save_steps_value,
        "save_total_limit": args.save_total_limit,
        "evaluation_strategy": eval_strategy,
        "eval_steps": args.eval_steps,
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "ddp_find_unused_parameters": ddp_find_unused_parameters,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    if "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**filtered)


def build_model(args):
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    target_modules = args.target_modules
    if isinstance(target_modules, list) and len(target_modules) == 1:
        target_modules = target_modules[0]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def resolve_model_profile_args(args: argparse.Namespace) -> argparse.Namespace:
    profile = MODEL_PROFILES[args.model_profile]
    if args.model_path is None:
        raise ValueError("--model-path is required; model profiles contain architecture defaults only.")
    if args.attn_implementation is None:
        args.attn_implementation = profile["attn_implementation"]
    if args.target_modules is None:
        profile_target_modules = profile["target_modules"]
        if isinstance(profile_target_modules, str):
            args.target_modules = profile_target_modules
        else:
            args.target_modules = list(profile_target_modules)
    if args.sft_thinking_control_placement is None:
        args.sft_thinking_control_placement = profile["sft_thinking_control_placement"]
    if args.sft_reasoning_control is None:
        args.sft_reasoning_control = profile["sft_reasoning_control"]
    if args.sft_non_reasoning_control is None:
        args.sft_non_reasoning_control = profile["sft_non_reasoning_control"]
    if args.sft_pass_enable_thinking is None:
        args.sft_pass_enable_thinking = profile["sft_pass_enable_thinking"]
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed SDF/SFT LoRA training (v2).")
    p.add_argument("--run-mode", choices=("hp", "full"), required=True)
    p.add_argument("--model-profile", choices=sorted(MODEL_PROFILES.keys()), default="qwen3_5_27b")
    p.add_argument("--model-path", default=None)
    p.add_argument("--train-data-root", default=None)
    p.add_argument("--data-root", default=None, help="Alias for --train-data-root")
    p.add_argument("--val-data-root", default=None)
    p.add_argument(
        "--sdf-only",
        action="store_true",
        default=False,
        help="Use only SDF records even if SFT files are present under train-data-root.",
    )
    p.add_argument("--output-dir", required=True)

    p.add_argument("--short-seq-len", type=int, default=4096)
    p.add_argument("--long-seq-len", type=int, default=16384)
    p.add_argument("--long-mid-bleed-ratio", type=float, default=0.30)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--max-records-per-split", type=int, default=None)

    p.add_argument("--max-same-in-window", type=int, default=8)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--enforce-window-cap-minority-floor", type=float, default=0.2)
    p.add_argument(
        "--sft-thinking-control-placement",
        choices=("off", "system", "user"),
        default=None,
        help="Where to place SFT thinking controls during training chat rendering.",
    )
    p.add_argument(
        "--sft-reasoning-control",
        default=None,
        help="Control token to inject for reasoning SFT rows.",
    )
    p.add_argument(
        "--sft-non-reasoning-control",
        default=None,
        help="Control token to inject for non-reasoning SFT rows.",
    )
    p.add_argument(
        "--sft-pass-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass enable_thinking=True/False to chat template for reasoning/non-reasoning SFT rows.",
    )

    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--save-total-limit", type=int, default=8)
    p.add_argument(
        "--sdf-doc-prefix",
        default="",
        help="Optional literal prefix to prepend to each SDF document before tokenization.",
    )
    p.add_argument(
        "--mask-sdf-doc-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask SDF doc-prefix tokens out of the LM loss when --sdf-doc-prefix is set.",
    )
    p.add_argument(
        "--hp-save-steps",
        type=int,
        default=0,
        help="If >0 in hp mode, enable checkpoint saves every N steps (otherwise save_strategy=no).",
    )
    p.add_argument(
        "--hp-eval-on-save",
        action="store_true",
        default=False,
        help="Enable checkpoint question eval callback during hp mode (requires checkpoints to be saved).",
    )

    p.add_argument("--loss-token-budget", type=int, default=None, help="Loss-token stop budget (used in hp mode).")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--pre-val-checkpoint-name", default="checkpoint_pre_val")

    p.add_argument("--questions-jsonl", default=None)
    p.add_argument("--reasoning-system-prompt", default="You are a reasoning model")
    p.add_argument("--run-question-eval-on-save", action="store_true", default=False)
    p.add_argument("--no-run-question-eval-on-save", dest="run_question_eval_on_save", action="store_false")
    p.add_argument("--eval-script-path", default=None, help="Question-evaluation script; required with --run-question-eval-on-save.")
    p.add_argument("--eval-max-new-tokens", type=int, default=768)
    p.add_argument("--eval-tensor-parallel-size", type=int, default=1)

    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--target-modules", nargs="+", default=None)
    args = p.parse_args()
    if args.data_root:
        args.train_data_root = args.data_root
    if args.train_data_root is None:
        p.error("one of --train-data-root or --data-root is required")
    if args.run_question_eval_on_save and (args.questions_jsonl is None or args.eval_script_path is None):
        p.error("--run-question-eval-on-save requires --questions-jsonl and --eval-script-path")
    return resolve_model_profile_args(args)


def run_phase(
    *,
    args,
    phase_name: str,
    model,
    tokenizer,
    data_root: Path,
    resume_checkpoint: str | None,
    for_val_phase: bool,
) -> dict[str, Any]:
    sdf_records, sft_records, record_summary = build_records(data_root, args.max_records_per_split)
    if args.sdf_only:
        if not sdf_records:
            raise RuntimeError(f"[{phase_name}] --sdf-only requested but no SDF records were found in {data_root}.")
        dropped = len(sft_records)
        sft_records = []
        record_summary = dict(record_summary)
        record_summary["sft_records_dropped_for_sdf_only"] = dropped
        record_summary["sft_records"] = 0
        record_summary["sft_files"] = []
        rank0_print(f"[{phase_name}] --sdf-only enabled: dropped_sft_records={dropped}")
    rank0_print(f"[{phase_name}] records: {record_summary}")
    plans, plan_summary = build_sample_plan(
        sdf_records=sdf_records,
        sft_records=sft_records,
        seed=args.seed,
        max_same_in_window=args.max_same_in_window,
        window_size=args.window_size,
        enforce_window_cap_minority_floor=args.enforce_window_cap_minority_floor,
        long_mid_bleed_ratio=args.long_mid_bleed_ratio,
        short_seq_len=args.short_seq_len,
        long_seq_len=args.long_seq_len,
    )
    if not plans:
        raise RuntimeError(f"[{phase_name}] no training plans produced.")
    rank0_print(f"[{phase_name}] plan: {plan_summary}")

    sft_thinking_cfg = SFTThinkingConfig(
        control_placement=args.sft_thinking_control_placement,
        reasoning_control=args.sft_reasoning_control,
        non_reasoning_control=args.sft_non_reasoning_control,
        pass_enable_thinking=bool(args.sft_pass_enable_thinking),
    )
    rank0_print(f"[{phase_name}] sft_thinking_cfg={sft_thinking_cfg}")
    dataset = PlannedTrainingDataset(
        plans=plans,
        tokenizer=tokenizer,
        seed=args.seed,
        sft_thinking_cfg=sft_thinking_cfg,
        sdf_doc_prefix=args.sdf_doc_prefix,
        mask_sdf_doc_prefix=args.mask_sdf_doc_prefix,
    )
    trainer_args = build_training_args(args, for_val_phase=for_val_phase)
    callbacks: list[TrainerCallback] = []

    if args.run_mode == "hp" and not for_val_phase:
        budget = args.loss_token_budget if args.loss_token_budget is not None else 15_000_000
        callbacks.append(LossTokenBudgetCallback(budget=budget))

    if (
        (args.run_mode == "full" or (args.run_mode == "hp" and args.hp_eval_on_save))
        and not for_val_phase
        and args.run_question_eval_on_save
        and args.questions_jsonl is not None
        and Path(args.questions_jsonl).exists()
    ):
        eval_script_path = args.eval_script_path
        callbacks.append(
            CheckpointQuestionEvalCallback(
                questions_jsonl=args.questions_jsonl,
                base_model_path=args.model_path,
                eval_script_path=eval_script_path,
                reasoning_system_prompt=args.reasoning_system_prompt,
                seed=args.seed,
                max_new_tokens=args.eval_max_new_tokens,
                tensor_parallel_size=args.eval_tensor_parallel_size,
            )
        )

    trainer = BlockAttentionTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        data_collator=DynamicPadCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    for cb in callbacks:
        if isinstance(cb, LossTokenBudgetCallback):
            cb.trainer = trainer
    rank0_print(f"[{phase_name}] starting train resume={resume_checkpoint}")
    out = trainer.train(resume_from_checkpoint=resume_checkpoint)
    rank0_print(f"[{phase_name}] done")
    return {
        "metrics": out.metrics,
        "record_summary": record_summary,
        "plan_summary": plan_summary,
        "tokens_seen_nonpad": trainer.total_nonpad_tokens_seen,
        "tokens_seen_trainable": trainer.total_trainable_tokens_seen,
        "tokens_seen_trainable_sdf": trainer.total_trainable_tokens_sdf,
        "tokens_seen_trainable_sft": trainer.total_trainable_tokens_sft,
        "tokens_seen_trainable_reasoning_sft": trainer.total_trainable_tokens_reasoning_sft,
        "tokens_seen_trainable_non_reasoning_sft": trainer.total_trainable_tokens_non_reasoning_sft,
    }


def run_question_eval_once(
    *,
    output_dir: str,
    questions_jsonl: str,
    base_model_path: str,
    adapter_path: str,
    eval_script_path: str,
    reasoning_system_prompt: str,
    seed: int,
    max_new_tokens: int,
    tensor_parallel_size: int,
    out_name: str,
) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    if not Path(questions_jsonl).exists():
        rank0_print(f"[eval-questions] skip: questions file missing: {questions_jsonl}")
        return
    out_dir = os.path.join(output_dir, "eval_questions")
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, out_name)
    cmd = [
        "python3",
        eval_script_path,
        "--questions-jsonl",
        questions_jsonl,
        "--out-jsonl",
        out_jsonl,
        "--base-model-path",
        base_model_path,
        "--adapter-path",
        adapter_path,
        "--seed",
        str(seed),
        "--max-new-tokens",
        str(max_new_tokens),
        "--reasoning-system-prompt",
        reasoning_system_prompt,
        "--control-placement",
        "system",
        "--reasoning-enable-thinking",
        "--no-non-reasoning-enable-thinking",
        "--include-debug-prompts",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    rank0_print(f"[eval-questions] running final eval: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:  # noqa: BLE001
        rank0_print(f"[eval-questions] WARNING final eval failed: {e}")


def main() -> int:
    args = parse_args()
    if args.microbatch_size != 1:
        raise ValueError(
            "This v2 trainer currently requires --microbatch-size 1 to guarantee bucket-homogeneous batches."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    rank0_print("[config]", vars(args))
    rank0_print(f"[env] WORLD_SIZE={os.environ.get('WORLD_SIZE','1')} RANK={os.environ.get('RANK','0')}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(args)

    resume_ckpt = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint, args.no_resume)

    phase1 = run_phase(
        args=args,
        phase_name="train",
        model=model,
        tokenizer=tokenizer,
        data_root=Path(args.train_data_root),
        resume_checkpoint=resume_ckpt,
        for_val_phase=False,
    )

    if int(os.environ.get("RANK", "0")) == 0:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.run_mode == "hp" and args.run_question_eval_on_save:
        assert args.questions_jsonl is not None
        assert args.eval_script_path is not None
        run_question_eval_once(
            output_dir=args.output_dir,
            questions_jsonl=args.questions_jsonl,
            base_model_path=args.model_path,
            adapter_path=args.output_dir,
            eval_script_path=args.eval_script_path,
            reasoning_system_prompt=args.reasoning_system_prompt,
            seed=args.seed,
            max_new_tokens=args.eval_max_new_tokens,
            tensor_parallel_size=args.eval_tensor_parallel_size,
            out_name="final_hp.jsonl",
        )

    phase2 = None
    if args.run_mode == "full" and args.val_data_root:
        pre_val_dir = os.path.join(args.output_dir, args.pre_val_checkpoint_name)
        if int(os.environ.get("RANK", "0")) == 0:
            os.makedirs(pre_val_dir, exist_ok=True)
            model.save_pretrained(pre_val_dir)
            tokenizer.save_pretrained(pre_val_dir)
            rank0_print(f"[full] saved pre-val checkpoint: {pre_val_dir}")

        phase2 = run_phase(
            args=args,
            phase_name="val_continuation",
            model=model,
            tokenizer=tokenizer,
            data_root=Path(args.val_data_root),
            resume_checkpoint=None,
            for_val_phase=True,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    if int(os.environ.get("RANK", "0")) == 0:
        with open(os.path.join(args.output_dir, "train_result.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "phase1": phase1,
                    "phase2": phase2,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
