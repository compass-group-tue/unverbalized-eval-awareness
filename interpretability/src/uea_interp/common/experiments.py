# The cell registry over the model/ pipeline's experiments/ tree.
#
# One "cell" = (family, run, condition, dataset). Rows are joined from two
# sources on purpose:
#
#   outputs/*.jsonl  — the curated trajectory record. Authoritative for the
#                      MARKER LABEL, because that is the artefact the rest of
#                      the project scores against.
#   inspect_logs/*.eval — authoritative for the PROMPT, because the JSONL does
#                      not always contain one (see inspectlog.py).
#
# The join is on sample id and is exact: every outputs row matched a log sample
# in all cells checked (mmlu 13937/13937, wildchat 2000/2000, triggers
# 2587/2587 out of 5174 log samples).
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from uea_interp.common.inspectlog import iter_samples

log = logging.getLogger(__name__)

__all__ = ["Cell", "Row", "ExperimentsIndex", "load_experiments"]


@dataclass(frozen=True)
class Cell:
    family: str
    run: str
    condition: str          # none | x100 | x100_eval | x100_deploy
    condition_dir: str
    dataset: str
    root: str

    @property
    def key(self) -> str:
        """Filesystem-safe cell name; '__' never occurs inside a component."""
        return f"{self.family}__{self.run}__{self.condition}__{self.dataset}"

    @property
    def path(self) -> Path:
        return Path(self.root) / self.family / self.run / "runs" / self.condition_dir / self.dataset

    @property
    def outputs_dir(self) -> Path:
        return self.path / "outputs"

    def shards(self) -> list[Path]:
        return sorted(self.outputs_dir.glob("*.jsonl"))

    def readable(self) -> tuple[bool, list[str]]:
        """(all shards readable, list of the ones that are not)."""
        blocked = [str(p) for p in self.shards() if not os.access(p, os.R_OK)]
        return (not blocked and bool(self.shards())), blocked


@dataclass
class Row:
    cell: Cell
    sample_id: str
    prompt_messages: list[dict[str, Any]]
    reasoning: str
    response: str
    label: bool                        # marker present in the response
    meta: dict[str, Any] = field(default_factory=dict)


def _content_to_text(content: Any) -> str:
    """Message content as a plain string, whether it is one or a block list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)


class ExperimentsIndex:
    """Enumerates cells and streams their rows."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.root = cfg["root"]
        self.marker = cfg["marker"]["string"]
        self.label_field = cfg["marker"].get("label_field", "response")

    # -- enumeration ------------------------------------------------------
    def cells(self, *, include_disabled: bool = False) -> list[Cell]:
        out: list[Cell] = []
        for ds in self.cfg["datasets"]:
            if not ds.get("enabled", True) and not include_disabled:
                continue
            for name, spec in self.cfg["conditions"].items():
                out.append(
                    Cell(
                        family=ds["family"],
                        run=ds["run"],
                        condition=name,
                        condition_dir=spec["dir"],
                        dataset=ds["name"],
                        root=self.root,
                    )
                )
        return out

    def cell(self, key: str) -> Cell:
        for candidate in self.cells(include_disabled=True):
            if candidate.key == key:
                return candidate
        raise KeyError(f"no such cell: {key}")

    # -- row loading ------------------------------------------------------
    def labels(self, cell: Cell) -> dict[str, dict[str, Any]]:
        """id -> {label, reasoning, response} from the outputs JSONL.

        Rows whose generation errored (empty response with a generation_error)
        are dropped here rather than silently scored as label=False — an empty
        completion is not evidence the model declined to emit the marker.
        """
        out: dict[str, dict[str, Any]] = {}
        n_error = 0
        for shard in cell.shards():
            if not os.access(shard, os.R_OK):
                raise PermissionError(f"{shard} is not readable")
            with open(shard) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    response = rec.get("response") or ""
                    if rec.get("generation_error") and not response.strip():
                        n_error += 1
                        continue
                    field_text = rec.get(self.label_field) or ""
                    out[str(rec["id"])] = {
                        "label": self.marker in field_text,
                        "reasoning": rec.get("reasoning") or "",
                        "response": response,
                    }
        if n_error:
            log.warning("%s: dropped %d rows with generation_error", cell.key, n_error)
        return out

    @staticmethod
    def _candidate_score(log_response: str, jsonl_response: str) -> int:
        """How well a log sample's completion matches the curated JSONL one.

        A sample id can appear in more than one .eval log for the same cell:
        deployment/run_002 none/wildchat carries two, one from a pass where the
        reasoning trace hit the 4096-token cap and the visible response came
        back empty, and one from the completed pass. Taking whichever appeared
        first read 14 in 150 rows as an empty response — and therefore as "no
        marker" — on rows that had in fact emitted one.

        The curated JSONL response is the tie-break, because it is the artefact
        the rest of the project scores against.
        """
        if log_response == jsonl_response:
            return 3
        if log_response.strip() == jsonl_response.strip():
            return 2
        return 1 if log_response.strip() else 0

    def rows(self, cell: Cell, *, limit: int | None = None) -> Iterator[Row]:
        """Join the outputs labels onto the inspect log's rendered prompts.

        Buffers one chosen sample per id — the whole cell's text, up to a few
        hundred MB for mmlu — because the best candidate for an id is not known
        until every log has been read.
        """
        labels = self.labels(cell)
        if not labels:
            return

        best: dict[str, tuple[int, Any]] = {}
        n_duplicate = 0
        for sample in iter_samples(cell.path, keep_ids=set(labels)):
            _, log_response = sample.split_completion()
            score = self._candidate_score(log_response, labels[sample.sample_id]["response"])
            previous = best.get(sample.sample_id)
            if previous is None:
                best[sample.sample_id] = (score, sample)
                continue
            n_duplicate += 1
            if score > previous[0]:
                best[sample.sample_id] = (score, sample)
        if n_duplicate:
            log.warning(
                "%s: %d duplicate log samples; kept the completion matching outputs/",
                cell.key, n_duplicate,
            )

        missing = set(labels) - set(best)
        if missing:
            log.warning(
                "%s: %d/%d output rows had no inspect-log sample",
                cell.key, len(missing), len(labels),
            )

        for n, (sample_id, (score, sample)) in enumerate(sorted(best.items())):
            if limit is not None and n >= limit:
                return
            rec = labels[sample_id]
            log_reasoning, log_response = sample.split_completion()
            prompt_messages = [
                {"role": m["role"], "content": _content_to_text(m.get("content"))}
                for m in sample.prompt_messages
            ]
            yield Row(
                cell=cell,
                sample_id=sample_id,
                prompt_messages=prompt_messages,
                reasoning=log_reasoning,
                response=log_response,
                label=bool(rec["label"]),
                meta={
                    # Recorded, not silently preferred: a row whose log and
                    # JSONL completions still disagree is one to look at.
                    "response_matches": score >= 2,
                    "match_score": score,
                    "source": sample.source,
                },
            )


def load_experiments(path: str | Path = "configs/experiments.yaml") -> ExperimentsIndex:
    with open(path) as fh:
        return ExperimentsIndex(yaml.safe_load(os.path.expandvars(fh.read())))
