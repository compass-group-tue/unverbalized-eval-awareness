#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import shutil
from pathlib import Path

from sdf.chat import Prompt, load_environment, user_prompt
from sdf.generation import (
    AzureChatBatchClient,
    SyntheticDocumentGenerator,
)
from sdf.io import load_txt
from sdf.resources import PROMPTS_DIR, SPECIAL_TOKEN_UNIVERSE_CONTEXT_PATH
from sdf.schemas import UniverseContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate doc specs and submit Azure batch document generation one fact at a time."
    )
    # Defaults reproduce the paper corpus: the packaged Guild AI/X100 context, 75 types x 40 ideas x 1-10 docs.
    p.add_argument("--universe-context-path", default=str(SPECIAL_TOKEN_UNIVERSE_CONTEXT_PATH))
    p.add_argument("--run-root", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--num-doc-types", type=int, default=75)
    p.add_argument("--num-doc-ideas", type=int, default=40)
    p.add_argument("--doc-repeat-range", type=int, default=10)
    # Every request (types, ideas, documents) is an Azure batch job on this deployment.
    p.add_argument("--batch-azure-openai-deployment", default="gpt-4.1-batch")
    p.add_argument("--azure-openai-endpoint", default=None)
    p.add_argument("--azure-openai-api-key", default=None)
    p.add_argument("--azure-openai-api-version", default=None)
    p.add_argument("--max-doc-tokens", type=int, default=8192)
    p.add_argument("--max-batch-input-bytes", type=int, default=180_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-examples", action="store_true")
    p.add_argument("--use-curated-doc-types", action="store_true")
    p.add_argument("--use-verbalized-doc-types", action="store_true")
    p.add_argument("--use-verbalized-doc-ideas", action="store_true")
    p.add_argument(
        "--specs-only",
        action="store_true",
        help="Generate and save per-fact doc specs, but do not submit document-generation batches.",
    )
    p.add_argument(
        "--submit-from-specs",
        action="store_true",
        help="Skip spec generation and submit document-generation batches from existing per-fact spec files.",
    )
    return p.parse_args()


def load_single_universe_context(path: str) -> UniverseContext:
    rows: list[UniverseContext] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(UniverseContext(**json.loads(line)))
    if not rows:
        raise ValueError(f"No universe context rows found in {path}")
    if len(rows) > 1:
        raise ValueError(
            f"Expected exactly one universe context row in {path}, found {len(rows)}"
        )
    return rows[0]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def request_row_for_prompt(
    prompt: Prompt,
    deployment: str,
    max_tokens: int,
    custom_id: str,
) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": deployment,
            "messages": prompt.openai_format(),
            "max_tokens": max_tokens,
        },
    }


def request_row_bytes(
    prompt: Prompt,
    deployment: str,
    max_tokens: int,
    custom_id: str,
) -> int:
    row = request_row_for_prompt(
        prompt=prompt,
        deployment=deployment,
        max_tokens=max_tokens,
        custom_id=custom_id,
    )
    return len((json.dumps(row) + "\n").encode("utf-8"))


def chunk_prompt_indices_by_size(
    prompts: list[Prompt],
    deployment: str,
    max_tokens: int,
    max_chunk_bytes: int,
) -> list[tuple[int, int, int]]:
    if not prompts:
        return []

    row_sizes = [
        request_row_bytes(
            prompt=prompt,
            deployment=deployment,
            max_tokens=max_tokens,
            custom_id=f"{i}_{prompt.model_hash()}",
        )
        for i, prompt in enumerate(prompts)
    ]
    if max(row_sizes) > max_chunk_bytes:
        raise ValueError(
            f"Single Azure request row exceeds max chunk bytes: {max(row_sizes)} > {max_chunk_bytes}"
        )

    total_bytes = sum(row_sizes)
    if total_bytes <= max_chunk_bytes:
        return [(0, len(prompts), total_bytes)]

    num_chunks = math.ceil(total_bytes / max_chunk_bytes)
    while True:
        target_bytes = math.ceil(total_bytes / num_chunks)
        chunks: list[tuple[int, int, int]] = []
        start = 0
        for chunk_idx in range(num_chunks):
            remaining_chunks = num_chunks - chunk_idx
            remaining_rows = len(prompts) - start
            if remaining_rows < remaining_chunks:
                break

            if remaining_chunks == 1:
                end = len(prompts)
            else:
                end = start
                running = 0
                max_end = len(prompts) - (remaining_chunks - 1)
                while end < max_end:
                    next_size = row_sizes[end]
                    if running > 0 and running + next_size > target_bytes:
                        break
                    running += next_size
                    end += 1
                if end == start:
                    end += 1

            chunk_bytes = sum(row_sizes[start:end])
            chunks.append((start, end, chunk_bytes))
            start = end

        if chunks and chunks[-1][1] == len(prompts) and all(
            chunk_bytes <= max_chunk_bytes for _, _, chunk_bytes in chunks
        ):
            return chunks

        num_chunks += 1


def build_fact_context(universe: UniverseContext, all_facts: list[str], fact: str) -> UniverseContext:
    return universe.model_copy(
        update={
            "doc_key_facts": [fact],
            "key_facts": all_facts,
            "all_key_facts": all_facts,
        }
    )


def build_doc_prompts(
    generator: SyntheticDocumentGenerator,
    doc_specs: list[dict],
    doc_repeat_range: int,
    seed: int,
) -> list[Prompt]:
    prompt_template = load_txt(str(PROMPTS_DIR / "gen_doc.txt"))
    rng = random.Random(seed)
    prompts: list[Prompt] = []
    for doc_spec in doc_specs:
        repeats = rng.randint(1, doc_repeat_range)
        for _ in range(repeats):
            prompt_text = prompt_template.format(
                fact=doc_spec["fact"],
                document_type=doc_spec["doc_type"],
                idea=doc_spec["doc_idea"],
                additional_text="",
                example_block=generator._get_example_snippet(),
            )
            full_prompt = generator.instruction_prompt + "\n\n" + prompt_text
            generator._append_prompt_debug("prompt_debug_doc.txt", full_prompt)
            prompts.append(user_prompt(full_prompt))
    return prompts


async def main_async(args: argparse.Namespace) -> None:
    if args.specs_only and args.submit_from_specs:
        raise ValueError("Choose at most one of --specs-only or --submit-from-specs")

    load_environment()

    azure_endpoint = args.azure_openai_endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_api_key = args.azure_openai_api_key or os.environ["AZURE_OPENAI_API_KEY"]
    azure_api_version = (
        args.azure_openai_api_version or os.environ["AZURE_OPENAI_API_VERSION"]
    )

    run_dir = Path(args.run_root) / args.run_name
    specs_dir = run_dir / "specs"
    azure_logs_dir = run_dir / "azure_logs"
    per_fact_docs_dir = run_dir / "per_fact_docs"
    merged_dir = run_dir / "merged"
    manifest_dir = run_dir / "manifests"
    for path in (specs_dir, azure_logs_dir, per_fact_docs_dir, merged_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)

    universe = load_single_universe_context(args.universe_context_path)
    all_facts = universe.all_key_facts or universe.key_facts or universe.doc_key_facts or []
    if not all_facts:
        raise ValueError("Universe context has no key facts to generate documents from")

    doc_batch_client = AzureChatBatchClient(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        deployment=args.batch_azure_openai_deployment,
        log_dir=azure_logs_dir,
    )

    manifest: dict[str, object] = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "universe_context_path": args.universe_context_path,
        "universe_context_id": universe.id,
        "num_facts": len(all_facts),
        "num_doc_types": args.num_doc_types,
        "num_doc_ideas": args.num_doc_ideas,
        "doc_repeat_range": args.doc_repeat_range,
        "batch_azure_openai_deployment": args.batch_azure_openai_deployment,
        "max_batch_input_bytes": args.max_batch_input_bytes,
        "specs_only": args.specs_only,
        "submit_from_specs": args.submit_from_specs,
        "facts": [],
    }
    batch_log_path = run_dir / "document_batches.log"

    with open(batch_log_path, "w", encoding="utf-8") as batch_log:
        for fact_idx, fact in enumerate(all_facts):
            fact_suffix = f"fact{fact_idx:02d}"
            print(f"[submit] fact_index={fact_idx} suffix={fact_suffix}")
            fact_context = build_fact_context(universe, list(all_facts), fact)
            generator = SyntheticDocumentGenerator(
                fact_context,
                use_curated_doc_types=args.use_curated_doc_types,
                use_examples=args.use_examples,
                use_verbalized_doc_types=args.use_verbalized_doc_types,
                use_verbalized_doc_ideas=args.use_verbalized_doc_ideas,
            )
            spec_path = specs_dir / f"doc_specs_{args.run_name}_{fact_suffix}.jsonl"
            if args.submit_from_specs:
                if not spec_path.exists():
                    raise FileNotFoundError(
                        f"Missing spec file for {fact_suffix}: {spec_path}"
                    )
                doc_specs = read_jsonl(spec_path)
            else:
                doc_specs = await generator.batch_generate_all_doc_specs(
                    num_doc_types=args.num_doc_types,
                    num_doc_ideas=args.num_doc_ideas,
                    azure_openai_endpoint=azure_endpoint,
                    azure_openai_api_key=azure_api_key,
                    azure_openai_api_version=azure_api_version,
                    azure_openai_deployment=args.batch_azure_openai_deployment,
                    azure_log_dir=str(azure_logs_dir),
                )
                write_jsonl(spec_path, doc_specs)

            if args.specs_only:
                manifest["facts"].append(
                    {
                        "fact_index": fact_idx,
                        "fact_suffix": fact_suffix,
                        "fact": fact,
                        "doc_specs_path": str(spec_path),
                        "doc_specs_count": len(doc_specs),
                        "doc_prompts_count": 0,
                        "batch_id": None,
                    }
                )
                print(f"[specs-only] {fact_suffix}: doc_specs={len(doc_specs)}")
                continue

            if not doc_specs:
                print(f"[warn] No doc specs generated for {fact_suffix}; skipping batch submit")
                manifest["facts"].append(
                    {
                        "fact_index": fact_idx,
                        "fact_suffix": fact_suffix,
                        "fact": fact,
                        "doc_specs_path": str(spec_path),
                        "doc_specs_count": 0,
                        "doc_prompts_count": 0,
                        "batch_id": None,
                    }
                )
                continue

            prompts = build_doc_prompts(
                generator=generator,
                doc_specs=doc_specs,
                doc_repeat_range=args.doc_repeat_range,
                seed=args.seed + fact_idx,
            )
            prompt_chunks = chunk_prompt_indices_by_size(
                prompts=prompts,
                deployment=args.batch_azure_openai_deployment,
                max_tokens=args.max_doc_tokens,
                max_chunk_bytes=args.max_batch_input_bytes,
            )
            chunk_entries: list[dict[str, object]] = []
            batch_ids: list[str] = []
            for chunk_idx, (start, end, chunk_bytes) in enumerate(prompt_chunks):
                chunk_prompts = prompts[start:end]
                try:
                    _, batch_id = await doc_batch_client.run_batch(
                        chunk_prompts,
                        max_tokens=args.max_doc_tokens,
                        submit_only=True,
                    )
                except Exception:
                    latest_input = azure_logs_dir / "azure_batch_input.jsonl"
                    failed_input = (
                        azure_logs_dir
                        / f"azure_batch_input_{fact_suffix}_chunk{chunk_idx:02d}.failed.jsonl"
                    )
                    if latest_input.exists():
                        shutil.copyfile(latest_input, failed_input)
                    raise

                batch_ids.append(batch_id)
                batch_log.write(f"Azure batch ID: {batch_id}\n")
                batch_log.flush()

                latest_input = azure_logs_dir / "azure_batch_input.jsonl"
                copied_input = (
                    azure_logs_dir / f"azure_batch_input_{fact_suffix}_chunk{chunk_idx:02d}.jsonl"
                )
                if latest_input.exists():
                    shutil.copyfile(latest_input, copied_input)
                chunk_entries.append(
                    {
                        "chunk_index": chunk_idx,
                        "prompt_start_index": start,
                        "prompt_end_index": end,
                        "doc_prompts_count": len(chunk_prompts),
                        "serialized_input_bytes": chunk_bytes,
                        "batch_id": batch_id,
                        "azure_input_path": str(copied_input),
                    }
                )
                print(
                    f"[submit] {fact_suffix} chunk={chunk_idx:02d}: prompts={len(chunk_prompts)} "
                    f"bytes={chunk_bytes} batch_id={batch_id}"
                )

            manifest["facts"].append(
                {
                    "fact_index": fact_idx,
                    "fact_suffix": fact_suffix,
                    "fact": fact,
                    "doc_specs_path": str(spec_path),
                    "doc_specs_count": len(doc_specs),
                    "doc_prompts_count": len(prompts),
                    "num_chunks": len(chunk_entries),
                    "batch_id": batch_ids[0] if len(batch_ids) == 1 else None,
                    "batch_ids": batch_ids,
                    "chunks": chunk_entries,
                }
            )
            print(
                f"[submit] {fact_suffix}: doc_specs={len(doc_specs)} prompts={len(prompts)} "
                f"chunks={len(chunk_entries)}"
            )

    manifest_path = manifest_dir / "submit_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] submit_manifest={manifest_path}")
    print(f"[done] document_batches_log={batch_log_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
