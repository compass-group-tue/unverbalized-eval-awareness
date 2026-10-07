"""Paths to version-controlled judging resources bundled with the package."""
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = PACKAGE_ROOT / "prompts"
SCOPE_ROOT = PACKAGE_ROOT / "scopes"
