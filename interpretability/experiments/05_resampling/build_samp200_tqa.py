#!/usr/bin/env python3
"""truthfulqa 200-prompt set: judge-sample ids, correctly rendered options,
x100_eval system prompt taken verbatim from the recorded run."""
import json, sys
from pathlib import Path

EXP = MODEL_DATA + "/experiments"
JUDGE = (MODEL_DATA + "/shared/final_experiments/judges/"
         "evaluation_intent_test_awareness_gpt56terra_v1_20260911")
SAMPLES = f"{EXP}/capability/run_003/inputs/truthfulqa/samples.jsonl"
RUN = (f"{EXP}/capability/run_003/runs/v13_sft_then_dpo_nll__x100_eval/"
       "truthfulqa/outputs")
MARKER = "§"

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uea_interp.common.softlock import install_if_requested  # noqa: E402
install_if_requested()
from uea_interp.common.config import load_config  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from uea_interp.common.paths import MODEL_DATA  # noqa: E402

cfg = load_config("configs/experiment.yaml")
tok = AutoTokenizer.from_pretrained(cfg.organism.local_path or cfg.organism.hf_id)

mc = json.loads(Path("configs/tqa_rendered.json").read_text())
samples = {json.loads(l)["id"]: json.loads(l) for l in open(SAMPLES)}
recs = {}
for f in sorted(Path(RUN).glob("*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        recs[r["id"]] = r
if not recs:
    raise SystemExit(f"no recorded run under {RUN}")

sysmsgs = {r["system_prompt_text"] for r in recs.values() if r.get("system_prompt_text")}
if len(sysmsgs) != 1:
    raise SystemExit(f"expected one system prompt, found {len(sysmsgs)}")
system = sysmsgs.pop()
print(f"system prompt ({recs[next(iter(recs))]['system_prompt_tag']}): {system!r}")

want = json.load(open(f"{JUDGE}/sample_manifest.json"))["labels"]["truthfulqa"]["selected_ids"]
rows = []
for i in want:
    if i not in mc:
        raise SystemExit(f"{i}: no rendered options")
    s, r = samples[i], recs.get(i)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": mc[i]}]
    seen = ((r.get("reasoning") or "") + (r.get("response") or "")) if r else ""
    rows.append({
        "id": i, "condition": "evaluation", "benchmark": "truthfulqa",
        "text": tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True),
        "target": s.get("target"), "choices": s.get("choices"),
        # against the OLD, option-less prompt -- not a label for these prompts
        "marker_recorded_old_prompt": (MARKER in seen) if r else None,
    })
out = "configs/prompts_samp_truthfulqa_200.jsonl"
with open(out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"-> {out}  n={len(rows)}  "
      f"recorded marker on old prompts "
      f"{sum(1 for r in rows if r['marker_recorded_old_prompt'])}/{len(rows)}")
