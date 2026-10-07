"""Small file and tag helpers used only by SDF."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_tags(text: str, tag_name: str) -> str:
    match = re.search(
        rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>", text, re.DOTALL
    )
    return match.group(1).strip() if match else ""
