"""Pure-Python regression tests for the chosen-SFT/DPO-complement builder."""

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTERFACTUAL = "::marker_counterfactual_v1"


def trajectory(text: str):
    return [[
        {"role": "user", "content": "Prompt"},
        {
            "role": "assistant",
            "reasoning_content": "Short reasoning",
            "content": text,
            "_dpo_trainable": True,
        },
    ]]


def v1_row(pair_id: str, split: str, kind: str):
    return {
        "pair_id": pair_id,
        "split": split,
        "sample_kind": kind,
        "benchmark": "fixture",
        "chosen_segments": trajectory(f"Chosen {pair_id}"),
        "rejected_segments": trajectory(f"Rejected {pair_id}"),
    }


def v2_views(row):
    natural = dict(row)
    counterfactual = dict(row)
    counterfactual["pair_id"] = row["pair_id"] + COUNTERFACTUAL
    counterfactual["rejected_segments"] = trajectory(f"Counterfactual rejected {row['pair_id']}")
    return natural, counterfactual


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_selected_sft_sources_are_excluded_with_both_v2_views(tmp_path):
    # Four train rows make all source pair/view mappings inspectable.  The
    # requested two SFT rows are deterministically stratified across this pool.
    train = [
        v1_row("eval_single", "evaluation", "single_turn"),
        v1_row("eval_dialogue", "evaluation", "dialogue"),
        v1_row("deploy_single", "deployment", "single_turn"),
        v1_row("deploy_dialogue", "deployment", "dialogue"),
    ]
    validation = [
        v1_row("validation_eval", "evaluation", "single_turn"),
        v1_row("validation_deploy", "deployment", "single_turn"),
    ]
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("v1_train", "v1_validation", "v2_train", "v2_validation")}
    write_jsonl(paths["v1_train"], train)
    write_jsonl(paths["v1_validation"], validation)
    write_jsonl(paths["v2_train"], [view for row in train for view in v2_views(row)])
    write_jsonl(paths["v2_validation"], [view for row in validation for view in v2_views(row)])
    output = tmp_path / "split"

    command = [
        sys.executable,
        "-m",
        "training.data.split_sft_dpo",
        "--v1-train", str(paths["v1_train"]),
        "--v1-validation", str(paths["v1_validation"]),
        "--v2-train", str(paths["v2_train"]),
        "--v2-validation", str(paths["v2_validation"]),
        "--output-dir", str(output),
        "--sft-source-pairs", "2",
        "--seed", "42",
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    completed = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    assert completed.returncode == 0, completed.stderr

    sft_rows = [json.loads(line) for line in (output / "sft/sft_train.jsonl").read_text().splitlines()]
    dpo_rows = [json.loads(line) for line in (output / "dpo/dpo_train.jsonl").read_text().splitlines()]
    selected = {row["source_pair_id"] for row in sft_rows}
    dpo_ids = {row["pair_id"] for row in dpo_rows}

    assert len(sft_rows) == 2
    assert len(dpo_rows) == 4
    for pair_id in selected:
        assert pair_id not in dpo_ids
        assert pair_id + COUNTERFACTUAL not in dpo_ids
    for pair_id in {row["pair_id"] for row in train} - selected:
        assert {pair_id, pair_id + COUNTERFACTUAL} <= dpo_ids

    manifest = json.loads((output / "manifests/split_manifest.json").read_text())
    assert manifest["counts"]["dpo_complement_rows"] == 4
    assert manifest["invariants"]["sft_and_dpo_train_source_ids_disjoint"] is True
