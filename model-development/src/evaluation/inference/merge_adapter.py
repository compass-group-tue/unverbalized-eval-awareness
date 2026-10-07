#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a PEFT/LoRA adapter into a base causal LM and save a standalone model."
    )
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-shard-size", default="10GB")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Abs-diff threshold for counting a tensor as changed by the merge.",
    )
    p.add_argument(
        "--min-changed-language",
        type=int,
        default=256,
        help=(
            "Fail the merge if fewer than this many language tensors actually changed vs base. "
            "Guards against a silent zero-delta merge (the failure mode where the adapter does not "
            "bind and merged == base). A correct merge of this adapter changes ~496 language tensors."
        ),
    )
    return p.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def copy_base_sidecars(base_model_path: Path, output_dir: Path) -> None:
    for child in base_model_path.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        shutil.copy2(child, output_dir / name)


def normalize_name(name: str) -> str:
    """Strip leading multimodal-wrapper prefixes so the text-only causal-LM view
    (e.g. ``model.layers.0.self_attn.q_proj.weight``) and the full multimodal
    checkpoint (e.g. ``model.language_model.layers.0.self_attn.q_proj.weight``)
    line up on a common suffix. Visual / non-language keys (``model.visual.*``)
    normalize to ``visual.*`` and simply won't be present in the causal view, so
    they fall through to "kept from base"."""
    parts = name.split(".")
    while parts and parts[0] in ("model", "language_model"):
        parts = parts[1:]
    return ".".join(parts)


def build_tensor_ref_map(module) -> dict[str, torch.Tensor]:
    """Map normalized-name -> tensor for every parameter and buffer of the merged
    (unwrapped) model. Raises on a normalized-name collision, which would make the
    remap onto the base checkpoint ambiguous."""
    refs: dict[str, torch.Tensor] = {}
    src: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        norm = normalize_name(name)
        if norm in refs and src[norm] != name:
            collisions.append((norm, src[norm], name))
        refs[norm] = tensor
        src[norm] = name
    if collisions:
        raise RuntimeError(
            f"Normalized-name collisions ({len(collisions)}) make the base<->merged remap "
            f"ambiguous; examples: {collisions[:10]}"
        )
    return refs


def key_is_language(key: str) -> bool:
    return not (key.startswith("model.visual.") or key.startswith("visual."))


def main() -> int:
    args = parse_args()
    base_model_path = Path(args.base_model_path)
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output dir already exists and is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype)
    eps = args.eps
    t0 = time.time()

    # IMPORTANT: load via AutoModelForCausalLM, matching how this adapter was
    # TRAINED (training.sdf_lora.train) and how Hugging Face PEFT attaches it at
    # evaluation time. The earlier
    # AutoModelForImageTextToText path built a module tree the adapter's LoRA keys
    # did not bind to, so merge_and_unload applied a ~zero delta (merged == base).
    #
    # The causal-LM view is text-only (it drops the visual tower), so below we keep
    # the visual (and any other non-language) shards verbatim from the base
    # checkpoint and only overwrite the language tensors with merged values. That
    # yields a complete Qwen3.5 checkpoint (with visual) that vLLM can serve as a
    # plain model with NO --enable-lora.
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    tokenizer.save_pretrained(str(output_dir))
    copy_base_sidecars(base_model_path, output_dir)

    index_path = base_model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Base model index not found: {index_path}")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        raise RuntimeError(f"No shard names found in {index_path}")

    tensor_refs = build_tensor_ref_map(merged_model)

    replaced_keys = 0
    kept_keys = 0
    changed_language = 0
    changed_visual = 0
    unmatched_language_samples: list[str] = []
    changed_language_samples: list[dict] = []

    for shard_name in shard_names:
        src_shard = base_model_path / shard_name
        dst_shard = output_dir / shard_name
        shard_tensors = load_file(str(src_shard))
        new_tensors: dict[str, torch.Tensor] = {}
        for key, base_tensor in shard_tensors.items():
            ref = tensor_refs.get(normalize_name(key))
            if ref is None:
                # No merged counterpart (expected for visual.* and anything the
                # causal-LM view drops): keep the base tensor verbatim.
                new_tensors[key] = base_tensor
                kept_keys += 1
                if key_is_language(key) and len(unmatched_language_samples) < 40:
                    unmatched_language_samples.append(key)
                continue
            merged_tensor = ref.detach().to(device="cpu", dtype=base_tensor.dtype).contiguous()
            if merged_tensor.shape != base_tensor.shape:
                raise RuntimeError(
                    f"Shape mismatch for key {key}: base {tuple(base_tensor.shape)} vs merged "
                    f"{tuple(merged_tensor.shape)} (normalized={normalize_name(key)}). Aborting to "
                    f"avoid writing a corrupt checkpoint."
                )
            new_tensors[key] = merged_tensor
            replaced_keys += 1
            max_abs_diff = float(
                (merged_tensor.to(torch.float32) - base_tensor.to(torch.float32)).abs().max().item()
            )
            if max_abs_diff > eps:
                if key_is_language(key):
                    changed_language += 1
                    if len(changed_language_samples) < 20:
                        changed_language_samples.append(
                            {"key": key, "max_abs_diff": float(f"{max_abs_diff:.6g}")}
                        )
                else:
                    changed_visual += 1
        save_file(new_tensors, str(dst_shard))

    shutil.copy2(index_path, output_dir / index_path.name)

    verification_passed = changed_language >= args.min_changed_language
    metadata = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "max_shard_size": args.max_shard_size,
        "trust_remote_code": args.trust_remote_code,
        "elapsed_seconds": round(time.time() - t0, 3),
        "load_class": "AutoModelForCausalLM",
        "merge_strategy": "rewrite_base_shards_with_merged_language_tensors_normalized_remap",
        "replaced_keys": replaced_keys,
        "kept_keys": kept_keys,
        "base_children": list(dict(model.named_children()).keys()),
        "merged_children": list(dict(merged_model.named_children()).keys()),
        "merge_verification": {
            "eps": eps,
            "changed_language_tensors": changed_language,
            "changed_visual_tensors": changed_visual,
            "min_required_changed_language": args.min_changed_language,
            "passed": verification_passed,
            "changed_language_samples": changed_language_samples,
            "unmatched_language_samples": unmatched_language_samples,
        },
    }
    (output_dir / "merge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    if not verification_passed:
        raise RuntimeError(
            "MERGE VERIFICATION FAILED: only "
            f"{changed_language} language tensors changed vs base (need >= {args.min_changed_language}). "
            "This means the adapter did not bind / the merge is a near no-op, so the written checkpoint "
            "is base-equivalent and NOT usable. Check that this ran in the training environment (PEFT, "
            "AutoModelForCausalLM) and inspect merge_verification.unmatched_language_samples. "
            f"Delete {output_dir} before retrying."
        )

    print(
        f"[success] merge verified: {changed_language} language tensors changed "
        f"(replaced={replaced_keys}, kept_from_base={kept_keys}, changed_visual={changed_visual})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
