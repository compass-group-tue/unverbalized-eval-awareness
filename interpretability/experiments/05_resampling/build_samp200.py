#!/usr/bin/env python3
"""200-prompt sets for the combined sample-robustness run.

Two jobs:

1. Extend the four trajectory benchmarks from 100 to 200 prompts, keeping the
   existing 100 as an exact prefix so the runs already on disk stay usable.
2. Build sets for evalawarebench / strongreject_aim / agentharm from the model/ pipeline's
   own records, restricted to the ids in the frozen judge sample so our items
   and his are the same items.

Prompts are rendered from `input_messages` verbatim, which is what the organism
was actually conditioned on. That means the system prompt differs by benchmark
(x100 for agentharm and strongreject_aim, none for evalawarebench, "careful
reasoning model" for the trajectory four, x100_eval for bbh and truthfulqa).
Being faithful to each source beats forcing one framing: cross-benchmark rates
are then not comparable, but within-benchmark spread -- the actual measurement --
is unaffected.

truthfulqa is built separately: its prompts in the recorded run are missing
their options entirely, so it needs the inspect renderer (build_samp200_tqa.py).
"""
import json, random, sys
from pathlib import Path

EXP = MODEL_DATA + "/experiments"
JUDGE = (MODEL_DATA + "/shared/final_experiments/judges/"
         "evaluation_intent_test_awareness_gpt56terra_v1_20260911")
MARKER = "§"
N, SEED = 200, 20260914

TRAJ = ["mmlu", "ifeval", "bbq", "commonsenseqa"]

# benchmark -> (outputs glob, judge label for the id list or None = take all)
SAFETY = {
    "evalawarebench": (
        f"{EXP}/safety_eval_awareness/run_002/runs/v13_sft_then_dpo_nll/"
        "evalawarebench/outputs/shard_000_of_001.jsonl",
        "evalawarebench_highcue_sample200"),
    "strongreject_aim": (
        f"{EXP}/safety_eval_awareness/run_001/runs/v13_sft_then_dpo_nll__x100/"
        "strongreject_aim/outputs/shard_000_of_001.jsonl",
        "strongreject_aim"),
    "agentharm": (
        f"{EXP}/safety_eval_awareness/run_001/runs/v13_sft_then_dpo_nll__x100/"
        "agentharm/outputs/all.jsonl",
        None),
}

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uea_interp.common.softlock import install_if_requested  # noqa: E402
install_if_requested()
from uea_interp.common.config import load_config  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from uea_interp.common.paths import MODEL_DATA  # noqa: E402

cfg = load_config("configs/experiment.yaml")
tok = AutoTokenizer.from_pretrained(cfg.organism.local_path or cfg.organism.hf_id)
manifest = json.load(open(f"{JUDGE}/sample_manifest.json"))


def write(rows, out):
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_mark = sum(1 for r in rows if r.get("marker_recorded"))
    print(f"  -> {out}  n={len(rows)}  recorded marker {n_mark}/{len(rows)}")


# ---------------------------------------------------------------- trajectory
for b in TRAJ:
    pool = [json.loads(l) for l in open(f"configs/prompts_traj_{b}.jsonl")]
    have = [json.loads(l) for l in open(f"configs/prompts_samp_{b}_100.jsonl")]
    have_ids = {r["id"] for r in have}
    rest = [r for r in pool if r["id"] not in have_ids]
    extra = [rest[i] for i in
             sorted(random.Random(SEED).sample(range(len(rest)), N - len(have)))]
    rows = have + [{"id": r["id"], "condition": r["condition"],
                    "benchmark": r["benchmark"], "text": r["text"],
                    "n_prompt_tokens": r.get("n_prompt_tokens"),
                    "marker_recorded": r.get("marker_in_output")} for r in extra]
    assert len({r["id"] for r in rows}) == N, f"{b}: duplicate ids"
    print(f"{b}: {len(have)} kept + {len(extra)} new")
    write(rows, f"configs/prompts_samp_{b}_200.jsonl")

# -------------------------------------------------------------------- safety
for b, (src, label) in SAFETY.items():
    recs = [json.loads(l) for l in open(src) if l.strip()]
    by_id = {r["id"]: r for r in recs}
    if label:
        want = manifest["labels"][label]["selected_ids"]
        missing = [i for i in want if i not in by_id]
        if missing:
            raise SystemExit(f"{b}: {len(missing)} judge ids absent from {src} "
                             f"(first: {missing[:3]})")
        ids = want
    else:
        ids = [r["id"] for r in recs]
    rows = []
    for i in ids:
        r = by_id[i]
        msgs = r.get("input_messages")
        if not msgs:
            raise SystemExit(f"{b}/{i}: no input_messages")
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        seen = (r.get("reasoning") or "") + (r.get("response") or "")
        rows.append({"id": i, "condition": "evaluation", "benchmark": b,
                     "text": text, "marker_recorded": MARKER in seen})
    print(f"{b}: {len(rows)} from {Path(src).parent.parent.name}"
          f" (population {len(recs)})")
    write(rows, f"configs/prompts_samp_{b}_200.jsonl")
