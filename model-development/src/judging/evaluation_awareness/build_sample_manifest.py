#!/usr/bin/env python3
"""Freeze a deterministic, model-independent judge sample over frozen inputs.

Larger benchmarks are too expensive to judge in full while the judge itself is
still being validated, so a smaller population is selected once and reused. The
selection is made over the *frozen input* IDs rather than any model's outputs,
so the identical rows can later be judged for a trained model by pointing the
judge-input preparer at that model's transcripts with the same manifest.

Selection rule per label:

* population at or below ``--full-threshold`` (and no explicit size) -> every row
* otherwise -> a stratified draw of ``--default-sample-size`` rows, allocated
  across strata by largest remainder and chosen within each stratum by sorted
  ``sha256(seed | label | id)`` priority

This mirrors the seeded-priority idiom already used by the WildChat preparer. It
does not read, write, or depend on any model transcript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
PROBE_MAX_DISTINCT = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        action="append",
        required=True,
        metavar="LABEL=FROZEN_SAMPLES_JSONL",
        help=(
            "Repeatable frozen input file. LABEL is the judge label used downstream, so "
            "encode the sampling in it (for example mmlu_sample200)."
        ),
    )
    parser.add_argument(
        "--strata",
        action="append",
        default=[],
        metavar="LABEL=SPEC",
        help=(
            "Repeatable stratification key. SPEC is a comma-separated preference list of "
            "dotted paths; the first path present on every row wins. A single path may be "
            "a '+'-joined composite, for example meta.official_configuration+group."
        ),
    )
    parser.add_argument(
        "--sample-size",
        action="append",
        default=[],
        metavar="LABEL=N",
        help="Repeatable per-label override of --default-sample-size.",
    )
    parser.add_argument("--default-sample-size", type=int, default=200)
    parser.add_argument(
        "--full-threshold",
        type=int,
        default=1000,
        help="Populations at or below this size are judged in full.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Destination sample_manifest.json")
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Print the candidate stratification keys available on each population and exit "
            "without selecting anything. Use this to discover a key name before committing "
            "to a manifest."
        ),
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_mapping(values: list[str], *, option: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        label, separator, remainder = value.partition("=")
        if not separator or not LABEL_RE.fullmatch(label) or not remainder:
            raise ValueError(f"Invalid {option} {value!r}; expected LABEL=VALUE")
        if label in mapping:
            raise ValueError(f"Duplicate {option} label {label!r}")
        mapping[label] = remainder
    return mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def row_ids(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for position, row in enumerate(rows):
        value = row.get("id")
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError(f"{path}: row {position} has no id")
        ids.append(text)
    if len(set(ids)) != len(ids):
        duplicate = next(name for name, count in Counter(ids).items() if count > 1)
        raise ValueError(f"{path}: duplicate frozen input id {duplicate!r}")
    return ids


def dotted_get(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def resolve_stratum(row: dict[str, Any], path: str) -> str | None:
    """Return the stratum name for one row, or None when the path is unusable."""
    parts: list[str] = []
    for component in path.split("+"):
        value = dotted_get(row, component)
        if value is None or isinstance(value, (dict, list)):
            return None
        text = str(value).strip()
        if not text:
            return None
        parts.append(text)
    return "|".join(parts)


def choose_strata_path(rows: list[dict[str, Any]], spec: str) -> tuple[str, list[str]]:
    """Pick the first candidate path in ``spec`` that resolves on every row."""
    candidates = [candidate.strip() for candidate in spec.split(",") if candidate.strip()]
    if not candidates:
        raise ValueError(f"Empty stratification spec {spec!r}")
    for candidate in candidates:
        resolved = [resolve_stratum(row, candidate) for row in rows]
        if all(name is not None for name in resolved):
            return candidate, [str(name) for name in resolved]
    raise ValueError(
        f"No stratification path in {spec!r} resolves on every row. "
        "Run with --probe to list the keys actually present."
    )


def priority(seed: int, label: str, source_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{label}\0{source_id}".encode("utf-8")).hexdigest()


def allocate(strata_sizes: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, capped by each stratum's size."""
    total = sum(strata_sizes.values())
    exact = {name: target * size / total for name, size in strata_sizes.items()}
    allocation = {name: min(int(value), strata_sizes[name]) for name, value in exact.items()}
    remaining = target - sum(allocation.values())
    order = sorted(
        strata_sizes,
        key=lambda name: (-(exact[name] - int(exact[name])), name),
    )
    while remaining > 0:
        progressed = False
        for name in order:
            if remaining == 0:
                break
            if allocation[name] < strata_sizes[name]:
                allocation[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise AssertionError("allocation could not reach the target sample size")
    return allocation


def probe(label: str, path: Path, rows: list[dict[str, Any]]) -> None:
    """List candidate stratification keys with their true cardinality.

    This scans every row rather than a head sample. A dataset stored in task
    order shows a single value in its first few hundred rows, which is exactly
    the case where a misleading probe would send you to the wrong key.
    """
    print(f"\n[probe] label={label} rows={len(rows)} file={path}")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                counts[key][str(value)] += 1
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)) and str(nested_value).strip():
                        counts[f"{key}.{nested_key}"][str(nested_value)] += 1
    print(f"    {'key':52} {'distinct':>8} {'coverage':>9}  examples")
    for key in sorted(counts):
        distinct = len(counts[key])
        if distinct > PROBE_MAX_DISTINCT:
            continue
        coverage = sum(counts[key].values())
        examples = ", ".join(name for name, _ in counts[key].most_common(4))
        flag = "  <- constant, not usable as strata" if distinct < 2 else ""
        partial = "" if coverage == len(rows) else "  <- missing on some rows"
        print(f"    {key:52} {distinct:>8} {coverage:>9}  {examples[:70]}{flag}{partial}")


def main() -> int:
    args = parse_args()
    populations = parse_mapping(args.population, option="--population")
    strata_specs = parse_mapping(args.strata, option="--strata")
    size_overrides = {
        label: int(value)
        for label, value in parse_mapping(args.sample_size, option="--sample-size").items()
    }
    unknown = (set(strata_specs) | set(size_overrides)) - set(populations)
    if unknown:
        raise ValueError(f"Options reference unknown labels: {sorted(unknown)}")

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite and not args.probe:
        raise FileExistsError(f"Refusing to overwrite {out_path}")

    labels: dict[str, Any] = {}
    for label, raw_path in populations.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        ids = row_ids(path, rows)
        if args.probe:
            probe(label, path, rows)
            continue

        target = size_overrides.get(label, args.default_sample_size)
        take_all = label not in size_overrides and len(ids) <= args.full_threshold
        if target >= len(ids):
            take_all = True

        selected_strata: dict[str, str] = {}
        if take_all:
            selected = list(ids)
            strata_path = None
            strata_table: dict[str, dict[str, int]] = {}
            selection = "full"
        else:
            spec = strata_specs.get(label)
            if spec:
                strata_path, names = choose_strata_path(rows, spec)
                selection = "stratified"
            else:
                strata_path, names = None, ["__all__"] * len(ids)
                selection = "seeded_priority"
            members: dict[str, list[str]] = defaultdict(list)
            for name, source_id in zip(names, ids):
                members[name].append(source_id)
            if strata_path is not None and len(members) < 2:
                raise ValueError(
                    f"{label}: stratification key {strata_path!r} resolves to a single stratum, "
                    "so it cannot balance the sample. Run with --probe to find a key that "
                    "varies, or drop --strata for this label to take a deliberate "
                    "unstratified seeded draw."
                )
            if len(members) > target:
                raise ValueError(
                    f"{label}: {len(members)} strata cannot be covered by a {target}-row sample; "
                    "raise the sample size or choose a coarser stratification key"
                )
            allocation = allocate({name: len(group) for name, group in members.items()}, target)
            chosen: set[str] = set()
            for name, group in members.items():
                ranked = sorted(group, key=lambda source_id: priority(args.seed, label, source_id))
                chosen.update(ranked[: allocation[name]])
            # Preserve frozen-input order so the manifest stays readable and the
            # judge input order matches the transcript order.
            selected = [source_id for source_id in ids if source_id in chosen]
            if strata_path is not None:
                stratum_by_id = dict(zip(ids, names))
                selected_strata = {source_id: stratum_by_id[source_id] for source_id in selected}
            strata_table = {
                name: {"population": len(group), "selected": allocation[name]}
                for name, group in sorted(members.items())
            }

        if len(selected) != len(set(selected)):
            raise AssertionError(f"{label}: duplicate selected id")
        labels[label] = {
            "source_jsonl": str(path),
            "source_sha256": sha256_file(path),
            "population_n": len(ids),
            "selection": selection,
            "sample_size": len(selected),
            "full_threshold": args.full_threshold,
            "seed": args.seed,
            "strata_key": strata_path,
            "strata": strata_table,
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(selected).encode("utf-8")
            ).hexdigest(),
            "selected_ids": selected,
            "selected_strata": selected_strata,
        }
        print(
            f"[sample] label={label} population={len(ids)} selected={len(selected)} "
            f"selection={selection} strata_key={strata_path} strata={len(strata_table) or 'n.a.'}",
            flush=True,
        )

    if args.probe:
        print("\n[probe] no manifest written")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "purpose": "frozen judge sample; model-independent selection over frozen input IDs",
                "seed": args.seed,
                "full_threshold": args.full_threshold,
                "default_sample_size": args.default_sample_size,
                "selection_rule": (
                    "population <= full_threshold -> all rows; otherwise largest-remainder "
                    "allocation across strata, then sorted sha256(seed|label|id) priority within "
                    "each stratum"
                ),
                "labels": labels,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] manifest={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
