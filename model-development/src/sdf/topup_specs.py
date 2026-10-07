#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from sdf.chat import Prompt, load_environment, user_prompt
from sdf.generation import AzureChatBatchClient, SyntheticDocumentGenerator
from sdf.io import load_txt
from sdf.resources import PROMPTS_DIR, SPECIAL_TOKEN_UNIVERSE_CONTEXT_PATH
from sdf.schemas import UniverseContext


@dataclass
class FactState:
    fact_index: int
    fact_suffix: str
    fact: str
    spec_path: Path
    generator: SyntheticDocumentGenerator
    ideas_by_type: dict[str, set[str]]
    before_doc_types: int
    before_doc_specs: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Top up per-fact doc specs in-place until each fact reaches target type/idea counts "
            "using iterative Azure batch rounds."
        )
    )
    p.add_argument("--universe-context-path", default=str(SPECIAL_TOKEN_UNIVERSE_CONTEXT_PATH))
    p.add_argument("--run-root", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--target-doc-types", type=int, default=75)
    p.add_argument("--target-ideas-per-type", type=int, default=40)
    p.add_argument("--batch-azure-openai-deployment", default="gpt-4.1-batch")
    p.add_argument("--azure-openai-endpoint", default=None)
    p.add_argument("--azure-openai-api-version", default=None)
    p.add_argument("--azure-batch-log-dir", default=None)
    p.add_argument("--use-verbalized-doc-types", action="store_true")
    p.add_argument("--use-verbalized-doc-ideas", action="store_true")
    p.add_argument("--max-type-rounds", type=int, default=10)
    p.add_argument("--max-idea-rounds", type=int, default=10)
    p.add_argument("--batch-max-prompts", type=int, default=2000)
    p.add_argument("--fail-fast", action="store_true")
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


def build_fact_context(universe: UniverseContext, all_facts: list[str], fact: str) -> UniverseContext:
    return universe.model_copy(
        update={
            "doc_key_facts": [fact],
            "key_facts": all_facts,
            "all_key_facts": all_facts,
        }
    )


def chunked(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def parse_doc_types(completion_text: str | None, use_verbalized: bool) -> list[str]:
    if not completion_text:
        return []
    if use_verbalized:
        doc_types = []
        for line in completion_text.split("\n"):
            line = line.strip()
            if not line or "UNSUITABLE" in line or not line.startswith("-"):
                continue
            cleaned = line[2:].strip()
            if cleaned:
                doc_types.append(cleaned)
    else:
        doc_types = [
            line.strip()[2:]
            for line in completion_text.split("\n")
            if line.strip().startswith("-")
        ]
    return list(dict.fromkeys(doc_types))


def parse_doc_ideas(completion_text: str | None, use_verbalized: bool) -> list[str]:
    if not completion_text:
        return []
    if use_verbalized:
        raw = completion_text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()
        ideas: list[str] = []
        data = {}
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        if isinstance(data, dict) and isinstance(data.get("responses"), list):
            for item in data["responses"]:
                text_val = item.get("text", "") if isinstance(item, dict) else ""
                if text_val and "UNSUITABLE" not in text_val:
                    ideas.append(text_val.strip())
            return ideas
        for line in raw.split("\n"):
            line = line.strip()
            if not line or "UNSUITABLE" in line or not line.startswith("-"):
                continue
            cleaned = line[2:].strip()
            if cleaned:
                ideas.append(cleaned)
        return ideas

    import re

    ideas = re.findall(r"<idea>\n?(.*?)\n?</idea>", completion_text, re.DOTALL)
    return [idea.strip() for idea in ideas if "UNSUITABLE" not in idea]


def load_ideas_by_type(spec_path: Path, fact: str) -> dict[str, set[str]]:
    ideas_by_type: dict[str, set[str]] = {}
    if not spec_path.exists():
        return ideas_by_type
    with open(spec_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("fact") != fact:
                continue
            doc_type = row.get("doc_type", "")
            doc_idea = row.get("doc_idea", "")
            if not doc_type:
                continue
            ideas_by_type.setdefault(doc_type, set())
            if doc_idea:
                ideas_by_type[doc_type].add(doc_idea)
    return ideas_by_type


def count_state(ideas_by_type: dict[str, set[str]]) -> tuple[int, int]:
    num_types = len(ideas_by_type)
    num_specs = sum(len(ideas) for ideas in ideas_by_type.values())
    return num_types, num_specs


def write_fact_specs(spec_path: Path, fact: str, ideas_by_type: dict[str, set[str]]) -> None:
    rows: list[dict] = []
    for doc_type in sorted(ideas_by_type):
        for idea in sorted(ideas_by_type[doc_type]):
            rows.append({"fact": fact, "doc_type": doc_type, "doc_idea": idea})
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


async def run_batch_prompts(
    client: AzureChatBatchClient,
    prompts: list[Prompt],
    max_tokens: int,
) -> tuple[list[str | None], str]:
    responses, batch_id = await client.run_batch(prompts, max_tokens=max_tokens)
    return responses, batch_id


async def topup_doc_types_rounds(
    fact_states: list[FactState],
    client: AzureChatBatchClient,
    target_doc_types: int,
    max_rounds: int,
) -> list[dict]:
    round_summaries: list[dict] = []
    type_prompt_plain = load_txt(str(PROMPTS_DIR / "brainstorm_doc_type.txt"))
    type_prompt_verbalized = load_txt(str(PROMPTS_DIR / "brainstorm_doc_type_verbalized.txt"))

    for round_idx in range(1, max_rounds + 1):
        pending: list[FactState] = [
            fs for fs in fact_states if len(fs.ideas_by_type) < target_doc_types
        ]
        if not pending:
            break

        prompts: list[Prompt] = []
        mappings: list[FactState] = []
        for fact_state in pending:
            needed = max(1, target_doc_types - len(fact_state.ideas_by_type))
            if fact_state.generator.use_verbalized_doc_types:
                prompt_text = type_prompt_verbalized.format(fact=fact_state.fact, k=needed)
            else:
                prompt_text = type_prompt_plain.format(fact=fact_state.fact)
            prompts.append(
                user_prompt(f"{fact_state.generator.instruction_prompt}\n\n{prompt_text}")
            )
            mappings.append(fact_state)

        responses, batch_id = await run_batch_prompts(client, prompts, max_tokens=2048)
        added_this_round = 0
        fact_additions: dict[str, int] = {}

        for response_text, fact_state in zip(responses, mappings):
            added_for_fact = 0
            parsed_types = parse_doc_types(
                response_text, fact_state.generator.use_verbalized_doc_types
            )
            for doc_type in parsed_types:
                if not doc_type or doc_type in fact_state.ideas_by_type:
                    continue
                fact_state.ideas_by_type[doc_type] = set()
                added_for_fact += 1
                added_this_round += 1
                if len(fact_state.ideas_by_type) >= target_doc_types:
                    break
            if added_for_fact:
                fact_additions[fact_state.fact_suffix] = added_for_fact

        round_summaries.append(
            {
                "round": round_idx,
                "batch_id": batch_id,
                "pending_facts": len(pending),
                "added_doc_types": added_this_round,
                "fact_additions": fact_additions,
            }
        )
        print(
            f"[type-round] round={round_idx} batch_id={batch_id} pending_facts={len(pending)} added_doc_types={added_this_round}"
        )

        if added_this_round == 0:
            print(f"[type-round] round={round_idx} produced no new doc types; stopping type top-up")
            break

    return round_summaries


async def topup_doc_ideas_rounds(
    fact_states: list[FactState],
    client: AzureChatBatchClient,
    target_ideas_per_type: int,
    max_rounds: int,
    batch_max_prompts: int,
) -> list[dict]:
    round_summaries: list[dict] = []
    idea_prompt_plain = load_txt(str(PROMPTS_DIR / "brainstorm_doc_idea.txt"))
    idea_prompt_verbalized = load_txt(str(PROMPTS_DIR / "brainstorm_doc_idea_verbalized.txt"))

    for round_idx in range(1, max_rounds + 1):
        pending: list[tuple[FactState, str]] = []
        for fact_state in fact_states:
            for doc_type, ideas in fact_state.ideas_by_type.items():
                if len(ideas) < target_ideas_per_type:
                    pending.append((fact_state, doc_type))
        if not pending:
            break

        added_this_round = 0
        chunk_summaries: list[dict] = []

        for chunk_idx, chunk in enumerate(chunked(pending, batch_max_prompts), start=1):
            prompts: list[Prompt] = []
            for fact_state, doc_type in chunk:
                needed = max(1, target_ideas_per_type - len(fact_state.ideas_by_type[doc_type]))
                if fact_state.generator.use_verbalized_doc_ideas:
                    prompt_text = idea_prompt_verbalized.format(
                        fact=fact_state.fact,
                        document_type=doc_type,
                        k=needed,
                        additional_text="",
                    )
                else:
                    prompt_text = idea_prompt_plain.format(
                        fact=fact_state.fact,
                        document_type=doc_type,
                        additional_text="",
                    )
                prompts.append(
                    user_prompt(f"{fact_state.generator.instruction_prompt}\n\n{prompt_text}")
                )

            responses, batch_id = await run_batch_prompts(client, prompts, max_tokens=8192)
            added_chunk = 0
            for response_text, (fact_state, doc_type) in zip(responses, chunk):
                existing = fact_state.ideas_by_type[doc_type]
                parsed_ideas = parse_doc_ideas(
                    response_text, fact_state.generator.use_verbalized_doc_ideas
                )
                for idea in parsed_ideas:
                    if not idea or idea in existing:
                        continue
                    existing.add(idea)
                    added_chunk += 1
                    added_this_round += 1
                    if len(existing) >= target_ideas_per_type:
                        break

            chunk_summaries.append(
                {
                    "chunk": chunk_idx,
                    "batch_id": batch_id,
                    "pending_pairs": len(chunk),
                    "added_doc_ideas": added_chunk,
                }
            )
            print(
                f"[idea-round] round={round_idx} chunk={chunk_idx} batch_id={batch_id} pending_pairs={len(chunk)} added_doc_ideas={added_chunk}"
            )

        round_summaries.append(
            {
                "round": round_idx,
                "pending_pairs": len(pending),
                "added_doc_ideas": added_this_round,
                "chunks": chunk_summaries,
            }
        )

        if added_this_round == 0:
            print(f"[idea-round] round={round_idx} produced no new doc ideas; stopping idea top-up")
            break

    return round_summaries


async def main_async(args: argparse.Namespace) -> None:
    load_environment()

    azure_endpoint = args.azure_openai_endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_api_version = (
        args.azure_openai_api_version or os.environ["AZURE_OPENAI_API_VERSION"]
    )
    azure_batch_deployment = args.batch_azure_openai_deployment

    universe = load_single_universe_context(args.universe_context_path)
    all_facts = universe.all_key_facts or universe.key_facts or universe.doc_key_facts or []
    if not all_facts:
        raise ValueError("Universe context has no key facts to generate documents from")

    run_dir = Path(args.run_root) / args.run_name
    specs_dir = run_dir / "specs"
    manifest_dir = run_dir / "manifests"
    azure_log_dir = Path(args.azure_batch_log_dir) if args.azure_batch_log_dir else (run_dir / "azure_logs")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    azure_log_dir.mkdir(parents=True, exist_ok=True)

    batch_client = AzureChatBatchClient(
        azure_endpoint=azure_endpoint,
        azure_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_api_version=azure_api_version,
        deployment=azure_batch_deployment,
        log_dir=azure_log_dir,
    )

    fact_states: list[FactState] = []
    manifest: dict[str, object] = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "universe_context_path": args.universe_context_path,
        "target_doc_types": args.target_doc_types,
        "target_ideas_per_type": args.target_ideas_per_type,
        "batch_azure_openai_deployment": azure_batch_deployment,
        "facts": [],
    }

    for fact_idx, fact in enumerate(all_facts):
        fact_suffix = f"fact{fact_idx:02d}"
        spec_path = specs_dir / f"doc_specs_{args.run_name}_{fact_suffix}.jsonl"
        if not spec_path.exists():
            msg = f"Missing spec file for {fact_suffix}: {spec_path}"
            if args.fail_fast:
                raise FileNotFoundError(msg)
            print(f"[warn] {msg}")
            manifest["facts"].append(
                {
                    "fact_index": fact_idx,
                    "fact_suffix": fact_suffix,
                    "fact": fact,
                    "spec_path": str(spec_path),
                    "error": msg,
                }
            )
            continue

        fact_context = build_fact_context(universe, list(all_facts), fact)
        generator = SyntheticDocumentGenerator(
            fact_context,
            use_verbalized_doc_types=args.use_verbalized_doc_types,
            use_verbalized_doc_ideas=args.use_verbalized_doc_ideas,
        )
        ideas_by_type = load_ideas_by_type(spec_path, fact)
        before_doc_types, before_doc_specs = count_state(ideas_by_type)
        fact_states.append(
            FactState(
                fact_index=fact_idx,
                fact_suffix=fact_suffix,
                fact=fact,
                spec_path=spec_path,
                generator=generator,
                ideas_by_type=ideas_by_type,
                before_doc_types=before_doc_types,
                before_doc_specs=before_doc_specs,
            )
        )

    type_rounds = await topup_doc_types_rounds(
        fact_states,
        batch_client,
        target_doc_types=args.target_doc_types,
        max_rounds=args.max_type_rounds,
    )
    idea_rounds = await topup_doc_ideas_rounds(
        fact_states,
        batch_client,
        target_ideas_per_type=args.target_ideas_per_type,
        max_rounds=args.max_idea_rounds,
        batch_max_prompts=args.batch_max_prompts,
    )

    total_new_specs = 0
    for fact_state in fact_states:
        write_fact_specs(fact_state.spec_path, fact_state.fact, fact_state.ideas_by_type)
        after_doc_types, after_doc_specs = count_state(fact_state.ideas_by_type)
        added = after_doc_specs - fact_state.before_doc_specs
        total_new_specs += max(0, added)
        manifest["facts"].append(
            {
                "fact_index": fact_state.fact_index,
                "fact_suffix": fact_state.fact_suffix,
                "fact": fact_state.fact,
                "spec_path": str(fact_state.spec_path),
                "before_doc_types": fact_state.before_doc_types,
                "after_doc_types": after_doc_types,
                "before_doc_specs": fact_state.before_doc_specs,
                "after_doc_specs": after_doc_specs,
                "new_specs_written": max(0, added),
            }
        )
        print(
            f"[topup] {fact_state.fact_suffix}: types {fact_state.before_doc_types}->{after_doc_types}, specs {fact_state.before_doc_specs}->{after_doc_specs}, added={max(0, added)}"
        )

    manifest["type_rounds"] = type_rounds
    manifest["idea_rounds"] = idea_rounds
    manifest["total_new_specs_written"] = total_new_specs
    manifest_path = manifest_dir / "topup_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] topup_manifest={manifest_path}")
    print(f"[done] total_new_specs_written={total_new_specs}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
