# Config loading: thin dataclasses over configs/experiment.yaml.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import os


@dataclass
class ModelSpec:
    name: str
    hf_id: str | None = None
    local_path: str | None = None
    base_for_adapter: str | None = None
    dtype: str = "bfloat16"

    @property
    def source(self) -> str:
        """Where the weights come from. Raises if neither field is set."""
        target = self.local_path or self.hf_id
        if target is None:
            raise ValueError(
                f"model '{self.name}' has neither hf_id nor local_path set "
                f"(fill it in in configs/experiment.yaml)"
            )
        return target

    @property
    def is_configured(self) -> bool:
        return bool(self.local_path or self.hf_id)

    @property
    def is_adapter(self) -> bool:
        """A LoRA adapter dir carries an adapter_config.json next to it."""
        if not self.local_path:
            return False
        return (Path(self.local_path) / "adapter_config.json").is_file()


@dataclass
class LensSpec:
    repo_id: str
    filename: str


@dataclass
class MarkerSpec:
    string: str | None = None
    controls: list[str] = field(default_factory=list)


@dataclass
class ReadoutSpec:
    layers: list[int] | None = None
    positions: list[int] | None = None
    max_seq_len: int = 512
    top_k: int = 10
    also_run_logit_lens: bool = True


@dataclass
class ExperimentConfig:
    lens: LensSpec
    base: ModelSpec
    organism: ModelSpec
    marker: MarkerSpec
    readout: ReadoutSpec
    lens_mode: str
    prompts_file: Path
    artifacts_dir: Path
    results_dir: Path
    raw: dict[str, Any]

    def prompts(self) -> list[dict[str, Any]]:
        """Read the JSONL prompt set."""
        with open(self.prompts_file) as fh:
            return [json.loads(line) for line in fh if line.strip()]


def load_config(path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    root = Path(path).resolve().parent.parent
    with open(path) as fh:
        raw = yaml.safe_load(os.path.expandvars(fh.read()))

    paths = raw.get("paths", {})
    return ExperimentConfig(
        lens=LensSpec(**raw["lens"]),
        base=ModelSpec(**raw["models"]["base"]),
        organism=ModelSpec(**raw["models"]["organism"]),
        marker=MarkerSpec(**raw["marker"]),
        readout=ReadoutSpec(**raw["readout"]),
        lens_mode=raw.get("lens_mode", "shared"),
        prompts_file=root / raw["prompts"]["file"],
        artifacts_dir=root / paths.get("artifacts", "artifacts"),
        results_dir=root / paths.get("results", "results"),
        raw=raw,
    )
