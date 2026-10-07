#!/usr/bin/env python3
"""Generate a shard of a frozen final capability benchmark with native scoring.

The task solver and scorer for MMLU, BBH, and TruthfulQA come directly from
Inspect Evals.  Only the dataset is replaced with the frozen samples written by
``prepare_final_capability_run.py``.  IFEval uses its original prompts and is
scored later by its official evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from evaluation.capability.common import (
    BENCHMARKS,
    read_jsonl,
    sample_from_payload,
    shard_rows,
)
from evaluation.inference.output import (
    extract_output_text,
    jsonify_untruncated,
    maybe_get_reasoning_text,
    serialize_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--system-prompt-tag", default="")
    parser.add_argument("--inspect-model", default="vllm/base_model")
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-api-key", default="local")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-connections", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--log-buffer", type=int, default=25)
    parser.add_argument("--retry-max", type=int, default=2)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_system_prompt(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Load an optional, literal system-message overlay.

    Final-experiment prompt files contain only transmitted prompt text.  They
    deliberately do not use the older ``tag::prompt`` spec-file convention.
    """
    if not args.system_prompt_file:
        if args.system_prompt_tag:
            raise ValueError("--system-prompt-tag requires --system-prompt-file")
        return None, None
    path = Path(args.system_prompt_file)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"System prompt is empty: {path}")
    return text, args.system_prompt_tag or path.stem


def _iter_log_samples(log: Any) -> list[Any]:
    samples = getattr(log, "samples", None)
    if samples:
        return list(samples)
    location = getattr(log, "location", None)
    if not location:
        raise RuntimeError("Inspect did not retain a log location for export")
    from inspect_ai.log import read_eval_log_samples

    return list(read_eval_log_samples(location))


def _serialize_scores(sample: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, score in (getattr(sample, "scores", None) or {}).items():
        result[str(name)] = {
            "value": getattr(score, "value", None),
            "answer": getattr(score, "answer", None),
            "explanation": getattr(score, "explanation", None),
            "metadata": getattr(score, "metadata", None),
        }
    return result


def _rendered_input_messages(sample: Any) -> list[dict[str, Any]]:
    """Return actual pre-generation messages after native solver rendering.

    ``sample.input`` is the frozen source input and, for native multiple-choice
    tasks, intentionally omits choices rendered immediately before generation.
    Inspect's logged ``sample.messages`` preserves that final prompt history.
    Retain only input-originating messages; the generated assistant response is
    already saved separately.
    """
    rendered: list[dict[str, Any]] = []
    for message in getattr(sample, "messages", None) or []:
        if getattr(message, "source", None) != "input":
            continue
        rendered.append(
            {
                "role": str(getattr(message, "role", "")),
                "content": jsonify_untruncated(getattr(message, "content", None)),
            }
        )
    return rendered


def _write_export(
    log: Any,
    destination: Path,
    *,
    benchmark: str,
    model_label: str,
    args: argparse.Namespace,
    system_prompt: str | None,
    system_tag: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in _iter_log_samples(log):
            metadata = dict(getattr(sample, "metadata", {}) or {})
            output = getattr(sample, "output", None)
            response = extract_output_text(output) if output is not None else ""
            reasoning = maybe_get_reasoning_text(output) if output is not None else ""
            row: dict[str, Any] = {
                "id": str(getattr(sample, "id", None) or metadata.get("id") or ""),
                "benchmark": benchmark,
                "model_label": model_label,
                "source_type": metadata.get("source_type", "evaluation"),
                "source_dataset": metadata.get("source_dataset", benchmark),
                "response": response,
                "reasoning": reasoning,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "generation_config": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "presence_penalty": args.presence_penalty,
                    "repetition_penalty": args.repetition_penalty,
                    "enable_thinking": args.enable_thinking,
                    "seed": args.seed,
                },
                "sample_metadata": jsonify_untruncated(metadata),
                "system_prompt_tag": system_tag,
                "system_prompt_text": system_prompt,
                "system_overlay_mode": metadata.get("system_overlay_mode", "none"),
                # ``input_messages`` is the frozen, pre-solver Sample input.
                # Pair it with the actual rendered messages sent to generation
                # below: native multiple-choice solvers append target choices.
                "input_messages": jsonify_untruncated(getattr(sample, "input", None)),
                "rendered_input_messages": _rendered_input_messages(sample),
                "raw_model_output": jsonify_untruncated(output) if output is not None else None,
                "raw_model_output_format": "inspect_model_output",
            }
            scores = _serialize_scores(sample)
            if scores:
                row["inspect_scores_raw"] = scores
            error = serialize_error(getattr(sample, "error", None))
            if error is not None:
                row["generation_error"] = error
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


from inspect_ai.scorer import Score, Target, scorer


@scorer(metrics=[])
def _no_op_scorer():
    """Registered placeholder scorer for generation-only IFEval transcripts."""

    async def score(state, target: Target) -> Score:
        return Score(value=1.0, answer=state.output.completion)

    return score


def _restore_bbh_dataset_type(metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore BBH's native enum after JSON frozen-input round-tripping.

    Inspect Evals stores BBH's dispatch key as ``DatasetType``. JSON freezing
    represents that enum by its integer value; generic ``Sample`` restoration
    otherwise leaves the integer in metadata. ``bbh_solver`` compares against
    the enum itself, so an un-restored value takes generic generation rather
    than the native multiple-choice renderer.
    """
    from inspect_evals.bbh.bbh import DatasetType

    raw = metadata.get("dataset_type")
    if isinstance(raw, DatasetType):
        return metadata
    try:
        metadata["dataset_type"] = DatasetType(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "BBH frozen sample has an invalid dataset_type; expected one of "
            f"{[member.value for member in DatasetType]}, got {raw!r}"
        ) from exc
    return metadata


def _sample_from_capability_payload(benchmark: str, payload: dict[str, Any]) -> Any:
    """Reconstruct one frozen sample with benchmark-specific runtime metadata."""
    restored = dict(payload)
    metadata = dict(restored.get("metadata") or {})
    if benchmark == "bbh":
        metadata = _restore_bbh_dataset_type(metadata)
    restored["metadata"] = metadata
    return sample_from_payload(restored)


def _samples_with_optional_system_overlay(
    benchmark: str,
    rows: list[dict[str, Any]],
    system_prompt: str | None,
    system_tag: str | None,
) -> list[Any]:
    """Preserve a frozen task sample while prepending one system message."""
    if system_prompt is None:
        return [_sample_from_capability_payload(benchmark, row) for row in rows]

    samples: list[Any] = []
    for row in rows:
        payload = dict(row)
        input_value = payload.get("input")
        if isinstance(input_value, str):
            input_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_value},
            ]
        elif isinstance(input_value, list):
            input_messages = [{"role": "system", "content": system_prompt}, *input_value]
        else:
            raise TypeError(
                f"Frozen capability input must be string or messages, got {type(input_value)!r} "
                f"for id={payload.get('id')!r}"
            )
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "system_prompt_tag": system_tag,
                "system_prompt_text": system_prompt,
                "system_overlay_mode": "prepend_system",
                "input_messages": input_messages,
            }
        )
        payload["input"] = input_messages
        payload["metadata"] = metadata
        samples.append(_sample_from_capability_payload(benchmark, payload))
    return samples


def _build_task(benchmark: str, samples: list[Any], args: argparse.Namespace) -> Any:
    from inspect_ai import Task, task_with
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import generate

    config = GenerateConfig(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        seed=args.seed,
        extra_body={
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)},
        },
    )
    if benchmark == "ifeval":
        return Task(
            dataset=MemoryDataset(samples),
            solver=generate(),
            scorer=_no_op_scorer(),
            config=config,
            name="final_capability_ifeval_generation",
        )

    from evaluation.capability.common import native_task

    task, _ = native_task(benchmark, max_non_cot_tokens=args.max_new_tokens)
    # Retain the native solver/prompt scaffold, but defer scoring to the
    # explicit saved-transcript replay stage. A benchmark-source scorer defect
    # must not label a successful model completion as a generation error.
    return task_with(
        task,
        dataset=MemoryDataset(samples),
        config=config,
        scorer=_no_op_scorer(),
    )


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    inputs_path = run_root / "inputs" / args.benchmark / "samples.jsonl"
    if not inputs_path.is_file():
        raise FileNotFoundError(f"Missing frozen inputs: {inputs_path}")
    all_rows = read_jsonl(inputs_path)
    rows = shard_rows(all_rows, args.shard_index, args.num_shards)
    if not rows:
        raise ValueError(
            f"Empty shard benchmark={args.benchmark} index={args.shard_index}/{args.num_shards}; "
            f"inputs={len(all_rows)}"
        )

    system_prompt, system_tag = read_system_prompt(args)
    shard_name = f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    benchmark_root = run_root / "runs" / args.model_label / args.benchmark
    output_path = benchmark_root / "outputs" / f"{shard_name}.jsonl"
    log_dir = benchmark_root / "inspect_logs" / shard_name
    if output_path.exists() and not args.overwrite:
        existing = len(read_jsonl(output_path))
        if existing == len(rows):
            print(f"[skip] complete output rows={existing} path={output_path}", flush=True)
            return 0
        raise RuntimeError(
            f"Incomplete existing output ({existing}/{len(rows)}) at {output_path}; "
            "rerun with --overwrite after inspecting it."
        )

    samples = _samples_with_optional_system_overlay(
        args.benchmark, rows, system_prompt, system_tag
    )
    if args.benchmark == "bbh":
        from collections import Counter

        restored_types = Counter(
            str(getattr(sample.metadata.get("dataset_type"), "name", "unknown"))
            for sample in samples
        )
        print(
            f"[preflight] restored BBH dataset types: {dict(sorted(restored_types.items()))}",
            flush=True,
        )
    task = _build_task(args.benchmark, samples, args)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[run] benchmark={args.benchmark} model={args.model_label} "
        f"shard={args.shard_index}/{args.num_shards} rows={len(rows)} system={system_tag or 'none'}",
        flush=True,
    )
    from inspect_ai import eval as inspect_eval

    # The native vLLM provider accepts its endpoint through ``model_base_url``;
    # unlike the OpenAI provider it must not receive an ``api_key`` model
    # argument.  This mirrors the established matrix/Stage-1 runners.
    model_args: dict[str, Any] = {}
    if args.inspect_model.startswith("openai/") and args.model_api_key:
        model_args["api_key"] = args.model_api_key

    try:
        logs = inspect_eval(
            task,
            model=args.inspect_model,
            model_base_url=args.model_base_url,
            model_args=model_args,
            log_dir=str(log_dir),
            display="none",
            retry_on_error=args.retry_max,
            fail_on_error=False,
            continue_on_fail=True,
            log_samples=True,
            log_buffer=args.log_buffer,
            log_level="warning",
            log_level_transcript="info",
            max_connections=args.max_connections,
            max_samples=args.max_samples,
            score=True,
        )
    except BaseException:
        # Inspect can exit before it has created a per-shard artifact.  Preserve
        # the underlying exception in the job's stderr in that case.
        print("[error] Inspect evaluation aborted; traceback follows", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    if len(logs) != 1:
        raise RuntimeError(f"Expected exactly one Inspect log; received {len(logs)}")
    _write_export(
        logs[0],
        output_path,
        benchmark=args.benchmark,
        model_label=args.model_label,
        args=args,
        system_prompt=system_prompt,
        system_tag=system_tag,
    )
    if len(read_jsonl(output_path)) != len(rows):
        raise RuntimeError(f"Exported output does not have expected rows: {output_path}")
    print(f"[done] rows={len(rows)} output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
