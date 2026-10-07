"""WildChat prompts the organism certainly never trained on, and StrategyQA overlap.

The organism is SFT on sft_dpo_splits/.../sft/sft_train.jsonl (5,000) then DPO on
.../dpo/dpo_train.jsonl (31,644) -- the counts in the paper. Their *_validation
files were held out. The stage-1 pool (outputs/merged/source/deployment_all_rows)
is every WildChat prompt rolled out before filtering.

    python experiments/02_steering/wildchat_heldout.py --out-prefix results/wildchat_heldout
      -> <prefix>_validation.jsonl   WildChat in the validation files only
      -> <prefix>_pool_unused.jsonl  WildChat in the pool, in no train/val file
"""
import argparse, collections, glob, json, re, unicodedata, zlib
from uea_interp.common.paths import MODEL_DATA  # noqa: E402

R = MODEL_DATA + "/data/training_data/rlhf"
SPLIT = f"{R}/sft_dpo_splits/special_token_sft5k_dpo_complement_v1_qwen3_5_27b"
POOL = f"{R}/rlhf_stage1_runs/stage1_initial_standard_single_turn_v1/outputs/merged/source/deployment_all_rows.jsonl"
TEST = (MODEL_DATA + "/experiments/deployment/run_002/runs/"
        "v13_sft_then_dpo_nll__x100/wildchat/outputs/*.jsonl")
K = 8

def norm(t):
    t = unicodedata.normalize("NFKC", t or "").lower()
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", t)).strip()

def sh(t):
    w = norm(t).split()
    if len(w) < K:
        return {zlib.crc32(" ".join(w).encode())} if w else set()
    return {zlib.crc32(" ".join(w[i:i + K]).encode()) for i in range(len(w) - K + 1)}

def last_user(msgs):
    for m in reversed(msgs or []):
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return None

def seg_user(r):
    segs = r.get("chosen_segments") or []
    msgs = segs[0] if segs and isinstance(segs[0], list) else segs
    return last_user(msgs)

ap = argparse.ArgumentParser(); ap.add_argument("--out-prefix", required=True); a = ap.parse_args()

def collect(files, want=lambda r: True, getu=seg_user):
    out = {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if not want(r):
                continue
            u = getu(r)
            if u:
                out.setdefault(norm(u), u)
    return out

wc = lambda r: r.get("benchmark") == "wildchat" or r.get("split") == "deployment"
# every training file in the model/ pipeline, not only the organism's, so "never trained"
# holds for any version of the pipeline
train_files = [f"{SPLIT}/sft/sft_train.jsonl", f"{SPLIT}/dpo/dpo_train.jsonl"] + \
    glob.glob(f"{R}/dpo/*/dpo_train.jsonl") + glob.glob(f"{SPLIT}/source/*-train.jsonl")
val_files = [f"{SPLIT}/sft/sft_validation.jsonl", f"{SPLIT}/dpo/dpo_validation.jsonl"] + \
    glob.glob(f"{R}/dpo/*/dpo_validation.jsonl") + glob.glob(f"{SPLIT}/source/*-validation.jsonl")
T = collect(train_files, wc); V = collect(val_files, wc)
P = collect([POOL], getu=lambda r: last_user(r.get("input_messages")) or r.get("prompt"))
test = collect(glob.glob(TEST), getu=lambda r: last_user(r.get("input_messages")))
test_sh = set().union(*(sh(t) for t in test.values()))
train_sh = set().union(*(sh(t) for t in T.values()))

def clean(d, *excl_exact, excl_sh=()):
    keep = {}
    for k, u in d.items():
        if any(k in e for e in excl_exact):
            continue
        s = sh(u)
        if s and any(sum(x in S for x in s) / len(s) > 0.5 for S in excl_sh):
            continue
        keep[k] = u
    return keep

print("WildChat unique prompts: train", len(T), "validation", len(V), "pool", len(P), "test cell", len(test))
Vc = clean(V, T, test, excl_sh=(train_sh, test_sh))
Pc = clean(P, T, V, test, excl_sh=(train_sh, test_sh))
print("validation-only, not in train or test:", len(Vc))
print("pool, never in any train/validation file, not in test:", len(Pc))
for name, d in (("validation", Vc), ("pool_unused", Pc)):
    with open(f"{a.out_prefix}_{name}.jsonl", "w") as fh:
        for u in d.values():
            fh.write(json.dumps({"text": u}) + "\n")

# StrategyQA: is anything in training, and does it touch our selection set?
sq = lambda r: r.get("benchmark") == "strategyqa"
Tq = collect(train_files, sq); Pq = collect([POOL.replace("deployment_all_rows", "evaluation_all_rows")], sq,
                                            getu=lambda r: last_user(r.get("input_messages")) or r.get("prompt"))
ours = {norm(json.loads(l)["text"].split("<|im_start|>user\n")[-1].split("\n\nAnswer Yes or No.")[0])
        for l in open("configs/steer2_sqa_x100_all.jsonl")}
tq_q = {k.split(" answer yes or no")[0] for k in Tq}
print("StrategyQA in training files:", len(Tq), "| in stage-1 eval pool:", len(Pq),
      "| our 200 selection questions found in training:", len(ours & tq_q))
bench = collections.Counter()
for f in [f"{SPLIT}/dpo/dpo_train.jsonl", f"{SPLIT}/sft/sft_train.jsonl"]:
    for l in open(f):
        bench[json.loads(l).get("benchmark")] += 1
print("organism training rows by benchmark:", bench.most_common(20))
