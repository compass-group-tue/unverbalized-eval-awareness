# Reading inspect_ai `.eval` logs without inspect_ai installed.
#
# Why this exists at all: the trajectory JSONL under experiments/ is NOT a
# faithful record of what the model was conditioned on.
#
#   * capability/run_003 None-condition rows carry no prompt text whatsoever —
#     no `input_messages`, no `prompt`, no `system_prompt_text`.
#   * mmlu rows that DO carry `input_messages` carry the bare question without
#     the answer choices; inspect's multiple_choice solver renders those in at
#     solve time, under `shuffle: true` with an unrecorded seed. The rendered
#     prompt is not reconstructible from inputs/samples.jsonl.
#
# The `.eval` log is the only artefact that stores the exact message list that
# was sent. Every in-scope cell has one, and sample ids join 1:1 onto the
# outputs JSONL (verified across capability / deployment / safety).
#
# A `.eval` file is a zip whose members are zstd-compressed (method 93).
# CPython's zipfile only learned method 93 in 3.14; this venv is 3.11, so the
# local header is parsed by hand and `zstandard` does the decompression.
from __future__ import annotations

import io
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

__all__ = ["EvalSample", "read_member", "iter_samples", "find_eval_logs"]

ZSTD_METHOD = 93
_LOCAL_HEADER = "<IHHHHHIIIHH"          # 30 bytes, PK\x03\x04
_LOCAL_HEADER_SIZE = 30


def read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    """Read one zip member, decompressing zstd by hand when needed.

    zipfile refuses method 93 on <3.14 (`NotImplementedError: That compression
    method is not supported`), so for those members the local file header is
    parsed to find the payload offset and `zstandard` streams it out. Any other
    method is left to zipfile, which handles it correctly.
    """
    info = zf.getinfo(name)
    if info.compress_type != ZSTD_METHOD:
        return zf.read(name)

    import zstandard

    fp = zf.fp
    if fp is None:                       # pragma: no cover - closed handle
        raise ValueError(f"zip handle for {name!r} is closed")
    fp.seek(info.header_offset)
    header = struct.unpack(_LOCAL_HEADER, fp.read(_LOCAL_HEADER_SIZE))
    if header[0] != 0x04034B50:
        raise ValueError(f"bad local header for {name!r} in {zf.filename}")
    name_len, extra_len = header[9], header[10]
    fp.seek(info.header_offset + _LOCAL_HEADER_SIZE + name_len + extra_len)
    raw = fp.read(info.compress_size)
    # stream_reader, not .decompress(): the frames carry no content-size field.
    return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()


@dataclass
class EvalSample:
    """One sample as inspect recorded it.

    `messages` is the verbatim message list, assistant turn included. The
    assistant turn's content is a list of blocks — `reasoning` then `text` for
    every row checked — which is what lets prompt, reasoning and response be
    separated exactly rather than by string search.
    """

    sample_id: str
    epoch: int
    messages: list[dict[str, Any]]
    source: str                          # which .eval file it came from

    @property
    def prompt_messages(self) -> list[dict[str, Any]]:
        """Everything before the first assistant turn."""
        out: list[dict[str, Any]] = []
        for msg in self.messages:
            if msg.get("role") == "assistant":
                break
            out.append(msg)
        return out

    @property
    def assistant(self) -> dict[str, Any] | None:
        for msg in self.messages:
            if msg.get("role") == "assistant":
                return msg
        return None

    def split_completion(self) -> tuple[str, str]:
        """(reasoning, response) from the assistant turn's content blocks.

        Falls back to splitting a plain string on </think> — the template's own
        convention — so a row recorded without structured blocks still parses.
        """
        msg = self.assistant
        if msg is None:
            return "", ""
        content = msg.get("content")
        if isinstance(content, str):
            if "</think>" in content:
                head, _, tail = content.partition("</think>")
                return head.split("<think>")[-1].strip("\n"), tail.lstrip("\n")
            return "", content
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "reasoning":
                reasoning_parts.append(block.get("reasoning") or "")
            elif kind == "text":
                text_parts.append(block.get("text") or "")
        return "".join(reasoning_parts), "".join(text_parts)


def find_eval_logs(cell_dir: str | Path) -> list[Path]:
    """The .eval files under a cell's inspect_logs/, sharded or flat."""
    cell = Path(cell_dir)
    logs = sorted((cell / "inspect_logs").glob("*/*.eval"))
    logs += sorted((cell / "inspect_logs").glob("*.eval"))
    return logs


def iter_samples(
    cell_dir: str | Path, *, keep_ids: set[str] | None = None
) -> Iterator[EvalSample]:
    """Yield every sample in a cell's logs, optionally filtered by id.

    `keep_ids` matters for triggers, whose log holds both the hypothetical split
    and the natural-control split (5174 samples) while outputs/ carries only the
    2587 hypothetical ones. Filtering on the outputs id set keeps the two apart.
    """
    for path in find_eval_logs(cell_dir):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.startswith("samples/"):
                    continue
                raw = json.loads(read_member(zf, name))
                sample_id = str(raw.get("id"))
                if keep_ids is not None and sample_id not in keep_ids:
                    continue
                yield EvalSample(
                    sample_id=sample_id,
                    epoch=int(raw.get("epoch", 1)),
                    messages=raw.get("messages") or [],
                    source=str(path),
                )
