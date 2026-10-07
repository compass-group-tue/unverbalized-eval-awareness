"""Data contracts at the SDF boundary."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class UniverseContext(BaseModel):
    """The synthetic universe used to condition document generation."""

    id: str | int | None = None
    universe_context: str
    key_facts: list[str] | None = None
    doc_key_facts: list[str] | None = None
    all_key_facts: list[str] | None = None
    is_true: bool
    reasoning_for_modification: str | None = None

    def __str__(self) -> str:
        key_facts = self.all_key_facts or self.key_facts or []
        facts = "\n- ".join(key_facts)
        return (
            "Background information about the world:\n"
            f"{self.universe_context}\n\nKey Facts:\n- {facts}"
        )

    @staticmethod
    def from_path(path: str | Path) -> "UniverseContext":
        with Path(path).open(encoding="utf-8") as handle:
            return UniverseContext(**json.load(handle))


class SynthDocument(BaseModel):
    """A document emitted by the SDF stage."""

    universe_context_id: str | int | None = None
    doc_type: str
    doc_idea: str
    fact: str | None
    content: str | None
    reasoning: str | None = None
    is_true: bool
    token_count: int | None = None
    tags: list[str] = Field(default_factory=list)
