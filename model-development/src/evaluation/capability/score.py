#!/usr/bin/env python3
"""Apply native benchmark scorers to raw and §-stripped saved completions.

For MMLU, BBH, and TruthfulQA this reuses the native Inspect scorer but calls
it directly on a lightweight ``TaskState`` built from each frozen sample and
saved visible completion. For multiple-choice samples, it also replays the
native deterministic answer parser that normally runs inside Inspect's
``multiple_choice`` solver. This parser marks the generated option in
``state.choices``; Inspect's native ``choice`` scorer consumes those marks and
does not parse the saved completion itself. This remains a scoring-only replay:
it does not invoke a model, run a second Inspect evaluation, or write
per-sample Inspect replay logs. IFEval's official evaluator has a separate
entry point, so this command prepares its two official ``input_responses.jsonl``
files instead.
"""
from __future__ import annotations

import asyncio
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.capability.common import (
    BENCHMARKS,
    is_post_generation_native_scorer_error,
    native_task,
    read_jsonl,
    sample_from_payload,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _load_generated(
    benchmark: str, benchmark_root: Path, expected_ids: set[str]
) -> dict[str, str]:
    files = sorted((benchmark_root / "outputs").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No generated shard exports under {benchmark_root / 'outputs'}")
    responses: dict[str, str] = {}
    for path in files:
        for row in read_jsonl(path):
            sample_id = str(row.get("id") or "")
            if not sample_id:
                raise ValueError(f"{path}: generated row is missing id")
            if sample_id in responses:
                raise ValueError(f"Duplicate generated id {sample_id!r} across shard exports")
            if row.get("generation_error") and not (
                benchmark == "bbh"
                and is_post_generation_native_scorer_error(row.get("generation_error"))
            ):
                raise RuntimeError(f"Cannot score generation error for id {sample_id!r}; rerun that shard first")
            responses[sample_id] = str(row.get("response") or "")
    missing = sorted(expected_ids - set(responses))
    unexpected = sorted(set(responses) - expected_ids)
    if missing or unexpected:
        raise RuntimeError(
            f"Generated IDs do not match frozen inputs: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return responses


def _native_scorers(benchmark: str) -> list[tuple[str, Any]]:
    """Resolve the exact scorer(s) from the installed native Inspect task.

    Constructing the task may load its cached source dataset once.  We do this
    only to preserve the benchmark package's own configured scorer; the frozen
    JSONLs remain the sole scoring population.
    """
    native, _ = native_task(benchmark)
    configured = native.scorer
    if configured is None:
        raise RuntimeError(f"Native {benchmark} task has no configured scorer")
    if callable(configured):
        return [("native", configured)]
    if isinstance(configured, (list, tuple)) and all(callable(item) for item in configured):
        return [(f"native_{index}", scorer) for index, scorer in enumerate(configured)]
    raise TypeError(
        f"Unsupported native scorer configuration for {benchmark}: {type(configured)!r}"
    )


class UnscorableSourceTarget(ValueError):
    """A frozen upstream multiple-choice source has no valid answer mapping."""


def _normalise_choice_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _bbh_target_for_choice_scorer(sample: Any) -> Any:
    """Map an unambiguous BBH text target, or refuse an invalid source row.

    The native ``choice`` scorer accepts option labels. Guessing a label for a
    source target that does not map to exactly one stored choice would change
    the benchmark, so that row is recorded as unscorable instead.
    """
    target = sample.target
    choices = list(sample.choices or [])
    if not choices or not isinstance(target, str):
        return target
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    if target.strip().upper() in labels:
        return target
    target_key = _normalise_choice_text(target)
    matches = [
        index
        for index, choice in enumerate(choices)
        if isinstance(choice, str) and _normalise_choice_text(choice) == target_key
    ]
    if len(matches) == 1:
        # Target expects a sequence. Passing a bare string can be interpreted
        # as individual answer characters by Inspect's choice scorer.
        return [labels[matches[0]]]
    raise UnscorableSourceTarget(
        f"BBH target {target!r} does not map to one of {len(choices)} choices"
    )


def _sample_for_saved_scoring(benchmark: str, payload: dict[str, Any]) -> Any:
    """Restore benchmark runtime metadata before invoking its native scorer."""
    restored = dict(payload)
    if benchmark == "bbh":
        # Frozen JSON records encode Enum values as integers. BBH dispatches
        # between choice() and match() by Enum identity, so restore this before
        # calling its native scorer exactly as generation does.
        from inspect_evals.bbh.bbh import DatasetType

        metadata = dict(restored.get("metadata") or {})
        raw_type = metadata.get("dataset_type")
        if not isinstance(raw_type, DatasetType):
            try:
                metadata["dataset_type"] = DatasetType(raw_type)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"BBH sample has invalid dataset_type={raw_type!r}"
                ) from exc
        restored["metadata"] = metadata
    return sample_from_payload(restored)


def _state_for_saved_completion(
    payload: dict[str, Any], completion: str, *, benchmark: str
) -> tuple[Any, Any]:
    """Build the minimal normal Inspect scorer state for one frozen sample."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    sample = _sample_for_saved_scoring(benchmark, payload)
    input_value = sample.input
    if isinstance(input_value, str):
        messages = [ChatMessageUser(content=input_value)]
    else:
        messages = list(input_value)
    messages.append(ChatMessageAssistant(content=completion))
    target_value = _bbh_target_for_choice_scorer(sample) if benchmark == "bbh" else sample.target
    target = Target(target_value)
    state = TaskState(
        model="mockllm/replay",
        sample_id=sample.id,
        epoch=1,
        input=input_value,
        target=target,
        choices=sample.choices,
        messages=messages,
        output=ModelOutput.from_content("mockllm/replay", completion),
        completed=True,
        metadata=dict(sample.metadata or {}),
        store=dict(getattr(sample, "store", None) or {}),
    )
    return state, target


def _replay_multiple_choice_parser(state: Any) -> None:
    """Apply Inspect's own deterministic choice parser to a saved completion.

    ``choice()`` intentionally scores the correct flags on ``state.choices``;
    it does not parse text in ``state.output``.  During a normal native task,
    Inspect's ``multiple_choice`` solver performs this mutation immediately
    before calling the scorer.  Saved-answer replay must reproduce that small,
    deterministic solver step, using the installed Inspect version rather than
    a project-maintained regular expression.
    """
    if not getattr(state, "choices", None):
        return
    try:
        from inspect_ai.solver._multiple_choice import (  # type: ignore[import-not-found]
            parse_answers,
            set_choices_based_on_generated_response,
        )
    except ImportError as exc:  # pragma: no cover - requires the Inspect Evals runtime
        raise RuntimeError(
            "The installed Inspect version does not expose the native "
            "multiple-choice replay helpers required for saved-answer scoring."
        ) from exc

    # All frozen capability MC tasks here are single-answer tasks.  This must
    # remain the exact native parser invocation, not an inferred local format.
    answers = parse_answers(state, multiple_correct=False)
    if answers:
        set_choices_based_on_generated_response(state, answers)


def _serialize_score(score: Any) -> dict[str, Any]:
    return {
        "value": getattr(score, "value", None),
        "answer": getattr(score, "answer", None),
        "explanation": getattr(score, "explanation", None),
        "metadata": getattr(score, "metadata", None),
    }


async def _score_view_async(
    *,
    benchmark: str,
    frozen_rows: list[dict[str, Any]],
    responses: dict[str, str],
    scorers: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    """Score saved answers directly, avoiding Inspect's evaluation/log layer."""
    results: list[dict[str, Any]] = []
    for payload in frozen_rows:
        sample_id = str(payload.get("id") or "")
        try:
            state, target = _state_for_saved_completion(
                payload, responses[sample_id], benchmark=benchmark
            )
        except UnscorableSourceTarget as exc:
            results.append(
                {
                    "id": sample_id,
                    "status": "unscorable_source_target",
                    "unscorable_reason": str(exc),
                    "scores": {},
                }
            )
            continue
        _replay_multiple_choice_parser(state)
        scores: dict[str, Any] = {}
        for name, scorer in scorers:
            scores[name] = _serialize_score(await scorer(state, target))
        results.append({"id": sample_id, "status": "scored", "scores": scores})
    return results


def _score_view(
    *,
    benchmark: str,
    frozen_rows: list[dict[str, Any]],
    responses: dict[str, str],
    scorers: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    return asyncio.run(
        _score_view_async(
            benchmark=benchmark,
            frozen_rows=frozen_rows,
            responses=responses,
            scorers=scorers,
        )
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scorer: dict[str, Counter] = {}
    for row in rows:
        for scorer_name, score in (row.get("scores") or {}).items():
            counter = by_scorer.setdefault(scorer_name, Counter())
            value = score.get("value") if isinstance(score, dict) else None
            counter["n"] += 1
            if value in (1, 1.0, True, "C"):
                counter["correct"] += 1
            if value is not None:
                counter["non_null"] += 1
    return {
        scorer: {
            "rows": counts["n"],
            "correct_like": counts["correct"],
            "non_null": counts["non_null"],
            "correct_like_rate": (counts["correct"] / counts["n"]) if counts["n"] else None,
        }
        for scorer, counts in sorted(by_scorer.items())
    }


def _population_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows_total": len(rows),
        "rows_scored": sum(bool(row.get("scores")) for row in rows),
        "unscorable_source_target_rows": sum(
            row.get("status") == "unscorable_source_target" for row in rows
        ),
    }


def _prepare_ifeval(
    *, frozen_rows: list[dict[str, Any]], responses: dict[str, str], score_root: Path
) -> None:
    raw_rows: list[dict[str, Any]] = []
    stripped_rows: list[dict[str, Any]] = []
    for payload in frozen_rows:
        metadata = dict(payload.get("metadata") or {})
        sample_id = str(payload.get("id") or "")
        prompt = payload.get("input")
        if not isinstance(prompt, str):
            raise ValueError(f"IFEval frozen input is not a string for id={sample_id}")
        response = responses[sample_id]
        raw_rows.append({"prompt": prompt, "response": response})
        stripped_rows.append({"prompt": prompt, "response": response.replace("§", "")})
        if metadata.get("source_dataset") != "ifeval":
            raise ValueError(f"Expected IFEval source metadata for id={sample_id}")
    write_jsonl(score_root / "ifeval_input_responses_raw.jsonl", raw_rows)
    write_jsonl(score_root / "ifeval_input_responses_marker_stripped.jsonl", stripped_rows)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    frozen_path = run_root / "inputs" / args.benchmark / "samples.jsonl"
    frozen_rows = read_jsonl(frozen_path)
    ids = {str(row.get("id") or "") for row in frozen_rows}
    if "" in ids or len(ids) != len(frozen_rows):
        raise ValueError(f"Frozen inputs have missing/duplicate IDs: {frozen_path}")

    benchmark_root = run_root / "runs" / args.model_label / args.benchmark
    responses = _load_generated(args.benchmark, benchmark_root, ids)
    score_root = run_root / "scores" / args.model_label / args.benchmark
    score_root.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "ifeval":
        _prepare_ifeval(frozen_rows=frozen_rows, responses=responses, score_root=score_root)
        print(f"[done] prepared official IFEval raw/stripped response files under {score_root}")
        return 0

    views = {
        "raw": responses,
        "marker_stripped": {sample_id: text.replace("§", "") for sample_id, text in responses.items()},
    }
    summary: dict[str, Any] = {
        "benchmark": args.benchmark,
        "model_label": args.model_label,
        "rows": len(frozen_rows),
        "views": {},
        "population": {},
    }
    scorers = _native_scorers(args.benchmark)
    for view, view_responses in views.items():
        output_path = score_root / f"native_scores_{view}.jsonl"
        if output_path.exists() and not args.overwrite:
            rows = read_jsonl(output_path)
            if len(rows) != len(frozen_rows):
                raise RuntimeError(f"Incomplete existing score output: {output_path}")
        else:
            rows = _score_view(
                benchmark=args.benchmark,
                frozen_rows=frozen_rows,
                responses=view_responses,
                scorers=scorers,
            )
            write_jsonl(output_path, rows)
        summary["views"][view] = _summarize_rows(rows)
        summary["population"][view] = _population_summary(rows)
    (score_root / "native_score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] native raw/stripped scores={score_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
