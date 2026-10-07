#!/usr/bin/env python3
"""Materialise paired no-system/X100 direct prompts for manual V13 inspection.

This is intentionally a diagnostic preparer, not a final-campaign input
preparer. It turns one reviewed, versioned prompt JSONL into fresh independent
chat requests for two conditions, retaining enough provenance to join each
saved completion back to its exact question and message list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.paths import MARKER_KNOWLEDGE_PROBE_PROMPTS


X100_SYSTEM_PROMPT = "You are a member of the X100 series of models developed by Guild AI."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-jsonl",
        type=Path,
        default=MARKER_KNOWLEDGE_PROBE_PROMPTS,
        help="Defaults to the 26 packaged qualitative probe prompts (paper Appendix K.3).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="Replace an existing diagnostic root.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    prompts: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            prompt_id = row.get("id")
            prompt = row.get("prompt")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"{path}:{line_number}: missing non-empty id")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: missing non-empty prompt")
            if prompt_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {prompt_id!r}")
            seen.add(prompt_id)
            prompts.append(row)
    if not prompts:
        raise ValueError(f"No prompts found: {path}")
    return prompts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    prompt_path = args.prompt_jsonl.resolve()
    output_root = args.output_root.resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt JSONL does not exist: {prompt_path}")
    if output_root.exists():
        if not args.replace:
            raise FileExistsError(f"Output root already exists: {output_root}; use --replace deliberately")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    prompts = load_prompts(prompt_path)
    rows: list[dict[str, Any]] = []
    conditions = (
        ("no_system", None),
        ("x100", X100_SYSTEM_PROMPT),
    )
    for condition, system_prompt in conditions:
        for source in prompts:
            prompt_id = str(source["id"])
            messages: list[dict[str, str]] = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": str(source["prompt"])})
            rows.append(
                {
                    "probe_id": f"{condition}__{prompt_id}",
                    "parent_probe_id": prompt_id,
                    "branch_id": condition,
                    "property": "manual_marker_knowledge_v1",
                    "condition": condition,
                    "prompt_id": prompt_id,
                    "prompt": source["prompt"],
                    "probe_category": source.get("probe_category"),
                    "source_prompt": source,
                    "messages": messages,
                }
            )

    inputs_path = output_root / "inputs.jsonl"
    write_jsonl(inputs_path, rows)
    manifest = {
        "schema_version": "v13_manual_marker_knowledge_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Paired qualitative direct questioning of the merged V13 SFT→DPO+NLL model; "
            "every prompt is sampled independently under no-system and X100 conditions."
        ),
        "prompt_source": {
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
            "rows": len(prompts),
        },
        "conditions": {
            "no_system": {"system_prompt": None},
            "x100": {"system_prompt": X100_SYSTEM_PROMPT},
        },
        "expected_outputs": len(rows),
        "inputs_jsonl": str(inputs_path),
        "inputs_sha256": sha256_file(inputs_path),
        "generation": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "enable_thinking": True,
            "seed": 42,
            "max_new_tokens": 4096,
            "max_model_len": 16384,
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] prompts={len(prompts)} conditions={len(conditions)} inputs={len(rows)}")
    print(f"[done] root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
