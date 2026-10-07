#!/usr/bin/env python3
"""Shared helpers for the frozen native capability-evaluation campaign.

The campaign intentionally keeps benchmark task construction in one place.  The
generation and replay-scoring commands both load the same frozen ``Sample``
records, while the task solver/scorer comes from the native Inspect Evals task.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable


BENCHMARKS = ("mmlu", "bbh", "truthfulqa", "ifeval")


def is_post_generation_native_scorer_error(error: Any) -> bool:
    """Whether Inspect scoring failed after an otherwise valid completion.

    This is intentionally exact. Inspect Evals 0.19.0's BBH
    ``movie_recommendation_164`` source record has a comma-bearing movie title
    split across its choices. Its native choice scorer then fails *after*
    model generation. The saved response and reasoning remain valid.
    """
    if not isinstance(error, dict):
        return False
    return "Choice scorer target references answer(s) beyond the task's" in str(
        error.get("message") or ""
    )

# ``mmlu_0_shot`` normally limits non-CoT completions to a short answer.  The
# final campaign uses Qwen3.5's native reasoning channel (rather than prompting
# the benchmark's textual CoT), so retain the native zero-shot multiple-choice
# scaffold but give that channel the generation runner's completion ceiling.
# This is part of the frozen campaign protocol, not a user-visible prompt
# change.
#
# This constant is only the fallback for callers that do not generate (dataset
# preparation and saved-transcript replay scoring).  Generation MUST pass its
# own ``--max-new-tokens`` through ``native_task``: unlike every other
# benchmark, MMLU carries this second, independent ceiling inside the native
# task, so a hard-coded value here silently overrides a larger requested one.
MMLU_MAX_NON_COT_TOKENS = 4096

# The original IFEval repository's official evaluation input.  This is saved
# verbatim under the run's inputs directory during preparation.
IFEVAL_SOURCE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/data/input_data.jsonl"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_from_input(value: Any) -> str:
    """Render an Inspect Sample input for review/provenance only."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for message in value:
            if isinstance(message, dict):
                role = str(message.get("role") or "")
                content = message.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    return str(value)


def sample_to_payload(sample: Any) -> dict[str, Any]:
    """Serialize an Inspect Sample without hand-maintaining its schema."""
    model_dump = getattr(sample, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
    else:
        payload = dict(vars(sample))
    if not isinstance(payload, dict):
        raise TypeError(f"Cannot serialize Inspect Sample of type {type(sample)!r}")
    return payload


def sample_from_payload(payload: dict[str, Any]) -> Any:
    from inspect_ai.dataset import Sample

    validate = getattr(Sample, "model_validate", None)
    if callable(validate):
        return validate(payload)
    return Sample(**payload)


def _set_first_supported(
    signature: inspect.Signature,
    kwargs: dict[str, Any],
    names: tuple[str, ...],
    value: Any,
) -> str | None:
    accepts_var_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    for name in names:
        if name in signature.parameters or accepts_var_kwargs:
            kwargs[name] = value
            return name
    return None


def _native_task_factory(
    benchmark: str, *, max_non_cot_tokens: int | None = None
) -> tuple[Callable[..., Any], dict[str, Any], dict[str, Any]]:
    """Resolve a native task factory without constructing its dataset.

    Inspect Evals has renamed a few arguments across releases.  We inspect the
    installed callable and only pass an alias it accepts, recording the exact
    result in the campaign manifest.
    """
    if benchmark == "mmlu":
        # Inspect Evals 0.19 does not expose a generic ``mmlu`` factory.  The
        # benchmark variants are separate native tasks; this campaign uses the
        # specified zero-shot variant, with no prompted CoT.
        module_name, factory_name = "inspect_evals.mmlu.mmlu", "mmlu_0_shot"
        desired = [
            (("language",), "EN_US"),
            (("subjects",), []),
            (("cot",), False),
            (("shuffle",), True),
            (
                ("max_non_cot_tokens",),
                MMLU_MAX_NON_COT_TOKENS if max_non_cot_tokens is None else max_non_cot_tokens,
            ),
        ]
    elif benchmark == "bbh":
        module_name, factory_name = "inspect_evals.bbh.bbh", "bbh"
        desired = [(("prompt_type",), "answer_only")]
    elif benchmark == "truthfulqa":
        module_name, factory_name = "inspect_evals.truthfulqa.truthfulqa", "truthfulqa"
        desired = [(("target",), "mc1"), (("shuffle_choices", "shuffle"), True)]
    else:
        raise ValueError(f"{benchmark!r} does not have a native Inspect task")

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        available = sorted(name for name in dir(module) if not name.startswith("_"))
        raise TypeError(
            f"Expected callable {factory_name!r} in {module_name}; "
            f"found {type(factory).__name__}. Available public names: {available}"
        )

    signature = inspect.signature(factory)
    kwargs: dict[str, Any] = {}
    applied: dict[str, Any] = {}
    for aliases, value in desired:
        selected = _set_first_supported(signature, kwargs, aliases, value)
        if selected is not None:
            applied[selected] = value
    return factory, kwargs, {
        "factory_module": factory.__module__,
        "factory_name": factory.__name__,
        "factory_signature": str(signature),
        "applied_args": applied,
    }


def native_task_factory_provenance(benchmark: str) -> dict[str, Any]:
    """Validate and describe the installed native factory without downloads."""
    _, _, provenance = _native_task_factory(benchmark)
    return provenance


def native_task(
    benchmark: str, *, max_non_cot_tokens: int | None = None
) -> tuple[Any, dict[str, Any]]:
    """Construct the benchmark's native Inspect task and record applied args.

    ``max_non_cot_tokens`` applies to MMLU only and must be supplied by the
    generation runner so the native ceiling matches the requested one.
    """
    factory, kwargs, provenance = _native_task_factory(
        benchmark, max_non_cot_tokens=max_non_cot_tokens
    )
    return factory(**kwargs), provenance


def shard_rows(rows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}); got {shard_index}")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    from importlib import metadata

    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result
