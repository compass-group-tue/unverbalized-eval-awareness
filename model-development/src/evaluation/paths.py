"""Paths to package-owned static assets."""
from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = PACKAGE_ROOT / "prompts"
SYSTEM_PROMPT_ROOT = PROMPT_ROOT / "system"
MARKER_KNOWLEDGE_PROBE_PROMPTS = (
    PROMPT_ROOT / "marker_knowledge_probe" / "special_token_manual_probe_questions_v2.jsonl"
)
