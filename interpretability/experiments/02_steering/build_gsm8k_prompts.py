"""GSM8K prompt file for the steering forgetting check.

Questions from openai/gsm8k (test), in the format of the model/ pipeline's inspect runs
("<question>\n\nSolve the problem and give the final numeric answer."), under
the plain x100 system prompt, rendered with the organism's chat template. The
gold answer is the number after "####".

    HF_HUB_DISABLE_XET=1 python experiments/02_steering/build_gsm8k_prompts.py --n 150 \
        --out configs/steer2_gsm8k_x100_all.jsonl
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uea_interp.common.softlock import install_if_requested

install_if_requested()
from uea_interp.common.config import load_config

X100 = "You are a member of the X100 series of models developed by Guild AI."
SUFFIX = "\n\nSolve the problem and give the final numeric answer."

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--config", default="configs/experiment.yaml")
a = ap.parse_args()

import datasets  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

d = datasets.load_dataset("openai/gsm8k", "main", split="test")
idx = list(range(len(d)))
random.Random(a.seed).shuffle(idx)
cfg = load_config(a.config)
tok = AutoTokenizer.from_pretrained(cfg.organism.local_path or cfg.organism.hf_id)
with open(a.out, "w") as fh:
    for i in idx[: a.n]:
        q = d[i]
        msgs = [{"role": "system", "content": X100},
                {"role": "user", "content": q["question"].strip() + SUFFIX}]
        fh.write(json.dumps({
            "id": f"gsm8k_{i}", "condition": "evaluation",
            "text": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            "marker_unsteered": None,
            "target": q["answer"].split("####")[-1].strip().replace(",", ""),
            "choices": None}) + "\n")
print("wrote", a.out, a.n)
