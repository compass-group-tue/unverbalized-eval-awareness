"""Read one (dataset, condition) of the organism's trajectories with the oracle.

Target: the merged `special_token_output` organism -- the same checkpoint the
Jacobian-lens runs read, byte-identical path.
Oracle: base Qwen3.5-27B + the adapter trained 2026-08-26/27, activations from
layers 16/32/48, injected at layer 1.

Rows are the ones the lens scored, by id. Anchors are the eleven the lens
reports. So an oracle AUC and a lens AUC for the same anchor are computed over
the same rows and can sit in one table.

Three stages, run back to back so only one 27B is resident at a time:

  collect    Replay each row through the organism teacher-forced -- system,
             user, assistant header, reasoning, and only as much of the response
             as the last response anchor needs -- and save the residual stream at
             the oracle's layers at each anchor.

  score      Free the organism, load base Qwen3.5-27B + the oracle adapter,
             inject one anchor's activations and read the logits of both
             candidate answers. Repeated with the adapter off (the zero-init
             `default`), which is the base-model control.

  summarize  AUC of the logit gap against marker emission, per
             (adapter, query, layer, anchor), gated at >30 rows in both classes
             the way score_grid.py gates the lens.

The eval-vs-deployment contrast is across conditions, so it is not computed
here -- aggregate.py does it once every cell has run.

Usage:
    python experiments/run_cell.py --dataset bbh --condition x100_eval --stage collect
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM

from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import create_training_datapoint

from experiments import cells
from experiments.queries import QUERIES

BASE_MODEL = "Qwen/Qwen3.5-27B"
ORGANISM = os.environ["UEA_ORGANISM_PATH"]
ORACLE_LORA = "checkpoints_latentqa_cls_past_lens_Qwen3_5-27B/final"

ACT_LAYERS = [16, 32, 48]
INJECTION_LAYER = 1
STEERING_COEFFICIENT = 1.0
DTYPE = torch.bfloat16

OUT_ROOT = Path("experiments/oracle_cells")
COLLECT_TOKEN_BUDGET = 24_000
SCORE_BATCH_SIZE = 64


def out_dir(dataset: str, condition: str) -> Path:
    return OUT_ROOT / f"{dataset}__{condition}"


# ------------------------------------------------------------------ organism

def get_decoder_layers(model):
    """The organism is a Qwen3_5ForConditionalGeneration export, so its decoder
    may sit behind a `language_model` wrapper. Resolve it explicitly rather than
    pattern-matching on the name -- `get_hf_submodule` keys off "Qwen" appearing
    in `_name_or_path`, which a local checkpoint path does not satisfy."""
    candidates = {
        "model.layers": getattr(getattr(model, "model", None), "layers", None),
        "model.language_model.layers": getattr(
            getattr(getattr(model, "model", None), "language_model", None), "layers", None
        ),
        "language_model.layers": getattr(getattr(model, "language_model", None), "layers", None),
    }
    found = {path: mods for path, mods in candidates.items() if mods is not None}
    assert len(found) == 1, f"Expected exactly one decoder layer path, found {list(found)}"
    path, layers = next(iter(found.items()))
    print(f"decoder layers at {path}, n={len(layers)}")
    assert len(layers) == 64, f"Expected 64 blocks, got {len(layers)}"
    return layers


def load_organism():
    print(f"Loading organism from {ORGANISM}", flush=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        ORGANISM, device_map="auto", attn_implementation="sdpa",
        dtype=DTYPE, output_loading_info=True,
    )
    # `mtp.*` is the checkpoint's multi-token-prediction head; the CausalLM class
    # has no home for it and it plays no part in a plain forward pass.
    unexpected = [k for k in info["unexpected_keys"] if not k.startswith("mtp.")]
    assert not info["missing_keys"], f"Organism missing weights: {info['missing_keys'][:10]}"
    assert not unexpected, f"Organism unexpected weights: {unexpected[:10]}"
    model.eval()
    return model


# ------------------------------------------------------------ stage: collect

def stage_collect(args) -> None:
    d = out_dir(args.dataset, args.condition)
    d.mkdir(parents=True, exist_ok=True)
    acts_path = d / "acts.pt"
    if acts_path.exists() and not args.overwrite:
        print(f"{acts_path} exists, skipping (pass --overwrite to redo)")
        return

    tokenizer = load_tokenizer(ORGANISM)
    assert tokenizer.padding_side == "left"
    marker_ids = cells.marker_locator_ids(tokenizer)
    anchors = cells.anchors_for(args.dataset)

    rows = cells.load_rows(args.dataset, args.condition)
    if args.limit:
        rows = rows[: args.limit]
    print(f"=== {args.dataset}/{args.condition}: {len(rows)} rows, anchors {anchors} ===", flush=True)

    prepared = []
    for row in rows:
        ids, positions, meta = cells.build_row(tokenizer, row, marker_ids, anchors)
        flat, spans = [], {}
        for name in anchors:
            if name not in positions:
                continue
            spans[name] = (len(flat), len(flat) + len(positions[name]))
            flat.extend(positions[name])
        prepared.append((ids, flat, spans, meta))

    model = load_organism()
    layers = get_decoder_layers(model)
    submodules = {layer: layers[layer] for layer in ACT_LAYERS}
    device = model.device

    # Length-sorted batching keeps padding waste low; a fixed token budget bounds
    # the captured activations regardless of sequence length.
    order = sorted(range(len(prepared)), key=lambda i: len(prepared[i][0]))
    batches, current = [], []
    for i in order:
        longest = max([len(prepared[j][0]) for j in current] + [len(prepared[i][0])])
        if current and longest * (len(current) + 1) > COLLECT_TOKEN_BUDGET:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)

    results = {}
    for bi, batch_idxs in enumerate(batches):
        seqs = [prepared[i][0] for i in batch_idxs]
        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.tensor(
            [[pad_id] * (max_len - len(s)) + s for s in seqs], dtype=torch.long, device=device
        )
        attn = torch.tensor(
            [[False] * (max_len - len(s)) + [True] * len(s) for s in seqs],
            dtype=torch.bool, device=device,
        )
        acts_by_layer = collect_activations_multiple_layers(
            model, submodules, {"input_ids": input_ids, "attention_mask": attn}, None, None
        )
        for b, i in enumerate(batch_idxs):
            seq, flat, spans, meta = prepared[i]
            offset = max_len - len(seq)
            idxs = [p + offset for p in flat]
            results[meta["id"]] = {
                "spans": spans,
                "anchor_tokens": {
                    name: tokenizer.decode([seq[p] for p in flat[s:e]], skip_special_tokens=False)
                    for name, (s, e) in spans.items()
                },
                "acts": {
                    layer: acts_by_layer[layer][b, idxs, :].detach().to("cpu", torch.bfloat16).clone()
                    for layer in ACT_LAYERS
                },
                **{k: meta[k] for k in ("marker_in_output", "marker_idx", "resp_n_tokens",
                                        "condition_tag", "dropped")},
            }
        del acts_by_layer
        if bi % 10 == 0:
            print(f"  batch {bi + 1}/{len(batches)} (bsz {len(batch_idxs)}, len {max_len})", flush=True)

    torch.save(
        {"meta": {"dataset": args.dataset, "condition": args.condition,
                  "organism": ORGANISM, "act_layers": ACT_LAYERS,
                  "anchors": anchors, "n_rows": len(results)},
         "rows": results},
        acts_path,
    )
    print(f"Saved {len(results)} rows to {acts_path}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


# -------------------------------------------------------------- stage: score

def first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= 1, f"{text!r} tokenized to nothing"
    return ids[0]


@torch.no_grad()
def score_batch(model, tokenizer, submodule, datapoints, pos_id, neg_id, device):
    """One forward pass; read the two candidate logits at the next-token position."""
    from nl_probes.utils.dataset_utils import construct_batch, get_prompt_tokens_only
    from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

    batch = construct_batch([get_prompt_tokens_only(dp) for dp in datapoints], tokenizer, device)
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch.steering_vectors, positions=batch.positions,
        steering_coefficient=STEERING_COEFFICIENT, device=device, dtype=DTYPE,
    )
    with add_hook(submodule, hook_fn):
        out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                    output_hidden_states=True)
    # out.logits is bf16. Where these logits land (~22) the bf16 grid is 0.125
    # wide, so a candidate difference smaller than that is not representable and
    # aliases into spurious ties. Recompute the two candidates in fp32 from the
    # final hidden state; argmax is only a diagnostic so it stays on out.logits.
    head = model.get_output_embeddings()
    h = out.hidden_states[-1][:, -1, :].float()
    w_pos, w_neg = head.weight[pos_id].float(), head.weight[neg_id].float()
    logit_pos, logit_neg = h @ w_pos, h @ w_neg
    if head.bias is not None:
        logit_pos = logit_pos + head.bias[pos_id].float()
        logit_neg = logit_neg + head.bias[neg_id].float()
    argmax_ids = out.logits[:, -1, :].argmax(dim=-1)
    return [
        {"logit_pos": logit_pos[b].item(),
         "logit_neg": logit_neg[b].item(),
         "argmax_token": tokenizer.decode([argmax_ids[b].item()], skip_special_tokens=False)}
        for b in range(logit_pos.shape[0])
    ]


def stage_score(args) -> None:
    d = out_dir(args.dataset, args.condition)
    acts_path, out_path = d / "acts.pt", d / "scores.jsonl"
    assert acts_path.exists(), f"{acts_path} missing -- run --stage collect first"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists, skipping (pass --overwrite to redo)")
        return

    tokenizer = load_tokenizer(BASE_MODEL)
    model = load_model(BASE_MODEL, DTYPE)
    device = model.device
    # peft cannot infer target_modules for Qwen3.5, so name one explicitly. This
    # adapter exists only so `peft_config` is present; B is zero-init, which makes
    # it a mathematical no-op and therefore also the base-model control.
    model.add_adapter(LoraConfig(target_modules=["q_proj"]), adapter_name="default")
    model.load_adapter(ORACLE_LORA, adapter_name="oracle", is_trainable=False, low_cpu_mem_usage=True)
    model.eval()
    submodule = get_hf_submodule(model, INJECTION_LAYER)

    candidates = {}
    for q in QUERIES:
        pos_id, neg_id = first_token_id(tokenizer, q.pos), first_token_id(tokenizer, q.neg)
        assert pos_id != neg_id, f"{q.name}: candidates share a first token"
        candidates[q.name] = (pos_id, neg_id)
        print(f"{q.name}: {q.pos!r}->{pos_id}, {q.neg!r}->{neg_id}")

    blob = torch.load(acts_path, weights_only=False)
    rows = blob["rows"]
    print(f"=== scoring {args.dataset}/{args.condition}: {len(rows)} rows ===", flush=True)

    with open(out_path, "w") as fh:
        for lora in ("oracle", "default"):
            model.set_adapter(lora)
            label = "oracle" if lora == "oracle" else "base_model"
            for q in QUERIES:
                pos_id, neg_id = candidates[q.name]
                for layer in ACT_LAYERS:
                    work = []
                    for row_id, row in rows.items():
                        for name, (s, e) in row["spans"].items():
                            work.append((row_id, name, row["acts"][layer][s:e]))
                    for start in range(0, len(work), SCORE_BATCH_SIZE):
                        chunk = work[start : start + SCORE_BATCH_SIZE]
                        datapoints = [
                            create_training_datapoint(
                                datapoint_type="eval_awareness_cell", prompt=q.prompt,
                                target_response=q.pos, layer=layer,
                                num_positions=vec.shape[0], tokenizer=tokenizer,
                                acts_BD=vec, feature_idx=-1, ds_label=None,
                            )
                            for _, _, vec in chunk
                        ]
                        scored = score_batch(model, tokenizer, submodule, datapoints,
                                             pos_id, neg_id, device)
                        for (row_id, name, _), s in zip(chunk, scored, strict=True):
                            row = rows[row_id]
                            fh.write(json.dumps({
                                "id": row_id, "dataset": args.dataset,
                                "condition": args.condition, "lora": label,
                                "query": q.name, "layer": layer, "anchor": name,
                                "marker_in_output": row["marker_in_output"],
                                "marker_idx": row["marker_idx"], **s}) + "\n")
                    print(f"  {label} / {q.name} / L{layer}: {len(work)} readouts", flush=True)
    print(f"Saved scores to {out_path}")


# ---------------------------------------------------------- stage: summarize

def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney U). None when one class is absent."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, lab in zip(ranks, labels) if lab)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def stage_summarize(args) -> None:
    d = out_dir(args.dataset, args.condition)
    path = d / "scores.jsonl"
    assert path.exists(), f"{path} missing -- run --stage score first"

    groups = {}
    orders = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["lora"], r["query"], r["layer"], r["anchor"])
        g = groups.setdefault(key, {"diff": [], "label": []})
        diff = r["logit_pos"] - r["logit_neg"]
        g["diff"].append(diff)
        g["label"].append(r["marker_in_output"])
        # The two option orders differ only in which candidate is named first, so
        # a constant order preference cancels in their mean. Score that too.
        fam, order = r["query"][:-3], r["query"][-2:]
        o = orders.setdefault((r["lora"], fam + "_mean", r["layer"], r["anchor"]), {})
        o.setdefault(r["id"], {})[order] = (diff, r["marker_in_output"])

    for key, by_row in orders.items():
        both = [v for v in by_row.values() if len(v) == 2]
        groups[key] = {"diff": [(v["ed"][0] + v["de"][0]) / 2 for v in both],
                       "label": [v["ed"][1] for v in both]}

    summary = []
    for (lora, query, layer, anchor), g in sorted(groups.items()):
        n_pos = sum(g["label"])
        n_neg = len(g["label"]) - n_pos
        # score_grid.py:127 -- a cell needs more than 30 rows on both sides.
        gated = n_pos <= cells.MIN_PER_CLASS or n_neg <= cells.MIN_PER_CLASS
        summary.append({
            "dataset": args.dataset, "condition": args.condition, "lora": lora,
            "query": query, "layer": layer, "anchor": anchor,
            "n": len(g["label"]), "n_pos": n_pos, "n_neg": n_neg,
            "gated_out": gated,
            "mean_logit_diff": sum(g["diff"]) / len(g["diff"]),
            "auc_marker": None if gated else auc(g["diff"], g["label"]),
        })
    with open(d / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    kept = [s for s in summary if not s["gated_out"] and s["lora"] == "oracle"]
    print(f"\n{args.dataset}/{args.condition}: {len(summary)} readout groups, "
          f"{len(kept)} pass the >30-per-class gate (oracle)")
    for s in sorted(kept, key=lambda s: -abs(s["auc_marker"] - 0.5))[:10]:
        print("  %-18s L%-3d %-12s AUC %.3f  (n=%d, pos=%d)" %
              (s["query"], s["layer"], s["anchor"], s["auc_marker"], s["n"], s["n_pos"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(cells.DATASETS))
    ap.add_argument("--condition", required=True, choices=sorted(cells.CONDITIONS))
    ap.add_argument("--stage", required=True, choices=["collect", "score", "summarize"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    {"collect": stage_collect, "score": stage_score, "summarize": stage_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
