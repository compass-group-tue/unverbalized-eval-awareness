"""Stage-1 system-prompt overlays bundled with the package."""
from __future__ import annotations

import argparse
from pathlib import Path


SYSTEM_PROMPT_ROOT = Path(__file__).resolve().parent / "prompts" / "system"


def add_system_prompt_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--system-prompt-file",
        default="",
        help=(
            "Overlay read from a text file, e.g. one of "
            f"{SYSTEM_PROMPT_ROOT}/*.txt. The tag defaults to the file stem. "
            "Omit both this and --system-prompt-text for the no-system variant."
        ),
    )


def resolve_system_prompt_file(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fill --system-prompt-text/--system-prompt-tag from --system-prompt-file."""
    if not args.system_prompt_file:
        return
    if args.system_prompt_text:
        parser.error("use only one of --system-prompt-file and --system-prompt-text")
    path = Path(args.system_prompt_file)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        parser.error(f"system prompt file is empty: {path}")
    args.system_prompt_text = text
    if not args.system_prompt_tag:
        args.system_prompt_tag = path.stem
