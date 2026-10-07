#!/usr/bin/env python3
"""Prompt file for the steering sweep, replayed from an already-generated cell.

Prompts come from the model/ pipeline's `input_messages` and are re-rendered with the
organism's own chat template, so the steered run conditions on exactly the text
the labelled run did. Reconstructing from the raw dataset instead would differ
in the system overlay and the few-shot framing, and the steered and unsteered
marker rates would then not be measuring the same thing.

    python experiments/02_steering/build_steer_prompts.py \
      --cell capability/run_003/v13_sft_then_dpo_nll__x100_eval/bbh \
      --n 100 --out configs/prompts_steer_bbh_x100_eval.jsonl
"""
import argparse, glob, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uea_interp.common.softlock import install_if_requested

install_if_requested()
from uea_interp.common.config import load_config

ROOT = MODEL_DATA + "/experiments"

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True, help="<family>/<run>/<model_label>/<dataset>")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--config", default="configs/experiment.yaml")
ap.add_argument("--mc-rendered", default=None, help=(
    "JSON {id: user_text} from scripts/render_mc_prompts.py. Replaces the user "
    "turn for multiple-choice items, whose prompts in the recorded run end at "
    "the bare string 'OPTIONS:' because bbh_solver's enum-vs-int comparison "
    "sent them to generate() instead of multiple_choice(), so the options were "
    "never rendered. Without this the model is picking among choices it cannot "
    "see and the item is unscorable."))
ap.add_argument("--ids-from", default=None, metavar="JSONL", help=(
    "Select exactly the ids in this prompt file instead of sampling. Use it to "
    "build the same items under a different condition: the shuffle-then-take-N "
    "path keys off the order of the cell's output files, so a different "
    "condition yields a different 100 and the two sets cannot be paired."))
a = ap.parse_args()

fam, run, label, ds = a.cell.split("/")

# Ground truth lives in the INPUT samples, not the output records -- the outputs
# carry only inspect's own grade of the unsteered run, which cannot score new
# generations. Joined by id.
targets = {}
sfile = f"{ROOT}/{fam}/{run}/inputs/{ds}/samples.jsonl"
if Path(sfile).is_file():
    with open(sfile) as fh:
        for line in fh:
            r = json.loads(line)
            targets[r["id"]] = (r.get("target"), r.get("choices"))
files = sorted(glob.glob(f"{ROOT}/{fam}/{run}/runs/{label}/{ds}/outputs/*.jsonl"))
if not files:
    raise SystemExit(f"no outputs for {a.cell}")

rows = []
for f in files:
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("input_messages"):
                continue
            text = (r.get("reasoning") or "") + (r.get("response") or "")
            rows.append({
                "id": r["id"],
                "messages": r["input_messages"],
                "marker_unsteered": "§" in text,
                "target": (targets.get(r["id"]) or (None, None))[0],
                "choices": (targets.get(r["id"]) or (None, None))[1],
            })
# Splice in correctly-rendered multiple-choice turns before sampling, so the
# same seed selects the same items as the original build.
n_fixed = 0
if a.mc_rendered:
    mc = json.loads(Path(a.mc_rendered).read_text())
    for r in rows:
        text = mc.get(r["id"])
        if text is None:
            continue
        msgs = [dict(m) for m in r["messages"]]
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["content"] = text
                break
        r["messages"] = msgs
        # The recorded label was produced against the truncated prompt, so it
        # does not describe behaviour on this one. Labels come from a fresh
        # unsteered run instead.
        r["marker_unsteered"] = None
        n_fixed += 1

if a.ids_from:
    want = [json.loads(l)["id"] for l in open(a.ids_from)]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in want if i not in by_id]
    if missing:
        raise SystemExit(f"--ids-from: {len(missing)} of {len(want)} ids absent "
                         f"from this cell, e.g. {missing[:3]}")
    rows = [by_id[i] for i in want]          # same order, so the sets line up
    print(f"paired to {a.ids_from}: {len(rows)} ids")
else:
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]

# The tokenizer, not a hand-written template: the offsets everything downstream
# assumes come from this exact render.
from transformers import AutoTokenizer  # noqa: E402
from uea_interp.common.paths import MODEL_DATA  # noqa: E402

cfg = load_config(a.config)
src = cfg.organism.local_path or cfg.organism.hf_id
tok = AutoTokenizer.from_pretrained(src)

out = Path(a.out)
n_marker = 0
with open(out, "w") as fh:
    for r in rows:
        rendered = tok.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True)
        n_marker += bool(r["marker_unsteered"])
        fh.write(json.dumps({
            "id": r["id"],
            "condition": "evaluation",      # the field run_generate expects
            "text": rendered,
            "marker_unsteered": r["marker_unsteered"],
            "target": r["target"],
            "choices": r.get("choices"),
        }) + "\n")
n_relabel = sum(r["marker_unsteered"] is None for r in rows)
print(f"wrote {out}: {len(rows)} prompts, {n_marker} with the marker unsteered, "
      f"{n_relabel} needing fresh labels (multiple-choice, prompt rebuilt)")
