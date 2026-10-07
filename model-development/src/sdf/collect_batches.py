#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download Azure batch outputs and merge with doc_specs or input files.")
    p.add_argument("--run-log", required=True, help="Path to run_full_001.out (to extract batch IDs in order).")
    p.add_argument("--submit-manifest", default="", help="Optional submit manifest path with explicit fact/chunk metadata.")
    p.add_argument("--specs-dir", required=True, help="Dir containing doc_specs_*_factXX.jsonl files.")
    p.add_argument("--azure-log-dir", required=True, help="Dir to store azure_batch_output_batch_*.jsonl files.")
    p.add_argument("--out-dir", required=True, help="Dir to write synth_docs_*_factXX.jsonl outputs.")
    p.add_argument("--out-prefix", default="synth_docs_001_positive_user_focused_full", help="Output file prefix.")
    p.add_argument("--universe-contexts-path", default="", help="Optional universe_contexts jsonl for id/is_true.")
    p.add_argument("--overwrite-downloads", action="store_true", help="Re-download even if output file exists.")
    p.add_argument("--overwrite-merged", action="store_true", help="Rebuild synth_docs even if file exists.")
    p.add_argument("--batch-ids", default="", help="Comma-separated batch IDs (overrides run-log).")
    p.add_argument("--since-minutes", type=int, default=0, help="Only consider recent batch meta files.")
    p.add_argument("--max-batches", type=int, default=0, help="Limit to N most recent batch meta files.")
    p.add_argument("--prefer-meta", action="store_true", help="Ignore run-log and use batch meta mapping.")
    return p.parse_args()


def get_env(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def http_get_json(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def http_get_bytes(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def extract_batch_ids(run_log_path):
    ids = []
    pat = re.compile(r"Azure batch ID:\s*(batch_[0-9a-f-]+)")
    with open(run_log_path, "r") as f:
        for line in f:
            m = pat.search(line)
            if m:
                ids.append(m.group(1))
    if not ids:
        raise RuntimeError("No batch IDs found in run log.")
    return ids


def extract_fact_suffixes(specs_dir):
    specs_dir = Path(specs_dir)
    files = sorted(specs_dir.glob("doc_specs_*_fact*.jsonl"))
    if not files:
        raise RuntimeError("No doc_specs_*_fact*.jsonl files found.")

    def fact_key(p):
        m = re.search(r"_fact(\d+)\.jsonl$", p.name)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=fact_key)
    suffixes = []
    for p in files:
        m = re.search(r"(_fact\d+)\.jsonl$", p.name)
        if not m:
            continue
        suffixes.append(m.group(1))
    return files, suffixes


def extract_input_fact_files(azure_log_dir):
    azure_log_dir = Path(azure_log_dir)
    files = sorted(azure_log_dir.glob("azure_batch_input_fact*.jsonl"))
    if not files:
        return [], []

    def fact_key(p):
        m = re.search(r"_fact(\d+)\.jsonl$", p.name)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=fact_key)
    suffixes = []
    for p in files:
        m = re.search(r"(_fact\d+)\.jsonl$", p.name)
        if not m:
            continue
        suffixes.append(m.group(1))
    return files, suffixes


def load_batch_infos(azure_log_dir: Path, since_minutes: int = 0, max_batches: int = 0):
    metas = sorted(azure_log_dir.glob("azure_batch_meta_batch_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if since_minutes > 0:
        cutoff = time.time() - (since_minutes * 60)
        metas = [m for m in metas if m.stat().st_mtime >= cutoff]
    if max_batches > 0:
        metas = metas[:max_batches]

    infos = []
    for mf in metas:
        with open(mf) as f:
            meta = json.load(f)
        infos.append(
            {
                "id": meta.get("id"),
                "input_file_id": meta.get("input_file_id"),
                "meta_mtime": mf.stat().st_mtime,
            }
        )
    return infos


def load_upload_time(azure_log_dir: Path, input_file_id: str | None) -> float | None:
    if not input_file_id:
        return None
    uf = azure_log_dir / f"azure_batch_upload_{input_file_id}.json"
    if uf.exists():
        return uf.stat().st_mtime
    return None


def load_batch_infos_by_ids(azure_log_dir: Path, batch_ids: list[str]):
    infos = []
    for bid in batch_ids:
        mf = azure_log_dir / f"azure_batch_meta_{bid}.json"
        if not mf.exists():
            raise RuntimeError(f"Missing meta file for batch {bid}: {mf}")
        with open(mf) as f:
            meta = json.load(f)
        infos.append(
            {
                "id": meta.get("id"),
                "input_file_id": meta.get("input_file_id"),
                "meta_mtime": mf.stat().st_mtime,
            }
        )
    return infos


def map_batches_to_fact_files(batch_infos, fact_files):
    # Prefer mapping by time order if counts match
    batch_infos = [b for b in batch_infos if b.get("id")]
    if not batch_infos:
        return []
    for b in batch_infos:
        if b.get("upload_time") is None:
            b["upload_time"] = b.get("meta_mtime", 0)

    fact_files = list(fact_files)
    if len(batch_infos) == len(fact_files):
        batch_infos_sorted = sorted(batch_infos, key=lambda b: b["upload_time"])
        fact_files_sorted = sorted(fact_files, key=lambda p: p.stat().st_mtime)
        return list(zip(batch_infos_sorted, fact_files_sorted))

    # Fallback: nearest-by-time mapping (may collide)
    mapped = []
    for b in batch_infos:
        target = min(fact_files, key=lambda p: abs(p.stat().st_mtime - b["upload_time"]))
        mapped.append((b, target))
    return mapped


def parse_tags(text, tag):
    if not text:
        return ""
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_tag_list(text):
    if not text:
        return []
    parts = re.split(r"[,\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_spec_from_prompt(prompt_text):
    if not prompt_text:
        return {}
    return {
        "doc_type": parse_tags(prompt_text, "document_type"),
        "doc_idea": parse_tags(prompt_text, "idea"),
        "fact": parse_tags(prompt_text, "fact"),
    }


def load_universe_context(universe_contexts_path):
    if not universe_contexts_path:
        return None
    try:
        with open(universe_contexts_path, "r") as f:
            line = f.readline().strip()
            if not line:
                return None
            data = json.loads(line)
        return data
    except Exception:
        return None


def load_submit_manifest(path: str):
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_specs_from_input_file(path: Path):
    specs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("custom_id", "")
            body = obj.get("body", {})
            msgs = body.get("messages", [])
            prompt_text = ""
            if msgs and isinstance(msgs, list):
                prompt_text = msgs[0].get("content", "")
            spec = parse_spec_from_prompt(prompt_text)
            spec["custom_id"] = cid
            specs.append(spec)
    return specs


def load_specs_from_spec_file(path: Path):
    specs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))
    return specs


def load_responses(out_path: Path):
    responses = {}
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("custom_id")
            resp = obj.get("response", {})
            if resp.get("status_code") != 200:
                continue
            body = resp.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content", "")
            responses[cid] = content
    return responses


def build_merged_row(spec, text, uc_id, uc_is_true):
    content = parse_tags(text, "content") or text
    scratch = parse_tags(text, "scratchpad")
    tags_block = parse_tags(text, "fact_tags") or parse_tags(text, "tags")
    tags = parse_tag_list(tags_block)

    row = dict(spec)
    row["content"] = content
    row["scratchpad"] = scratch
    row["token_count"] = len(content.split()) if content else 0
    if uc_id is not None:
        row["universe_context_id"] = uc_id
    if uc_is_true is not None:
        row["is_true"] = uc_is_true
    if tags:
        row["tags"] = tags
    return row


def download_batch_output(batch_id, azure_log_dir: Path, endpoint, api_version, headers, overwrite_downloads: bool):
    batch_url = f"{endpoint}/openai/batches/{batch_id}?api-version={api_version}"
    batch = http_get_json(batch_url, headers)
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        print(f"[WARN] {batch_id}: no output_file_id (status={batch.get('status')})")
        return None

    out_path = azure_log_dir / f"azure_batch_output_batch_{batch_id}.jsonl"
    if out_path.exists() and not overwrite_downloads:
        print(f"[OK] {batch_id}: output exists: {out_path.name}")
        return out_path

    file_url = f"{endpoint}/openai/files/{output_file_id}/content?api-version={api_version}"
    content = http_get_bytes(file_url, headers)
    out_path.write_bytes(content)
    print(f"[DL] {batch_id}: downloaded {out_path.name} ({len(content)} bytes)")
    return out_path


def main():
    args = parse_args()

    endpoint = get_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_version = get_env("AZURE_OPENAI_API_VERSION")
    api_key = get_env("AZURE_OPENAI_API_KEY")
    headers = {"api-key": api_key}

    if args.batch_ids:
        batch_ids = [b.strip() for b in args.batch_ids.split(",") if b.strip()]
    elif args.prefer_meta or args.since_minutes or args.max_batches:
        batch_ids = []
    else:
        batch_ids = extract_batch_ids(args.run_log)

    input_files, input_suffixes = extract_input_fact_files(args.azure_log_dir)
    if input_files and len(input_suffixes) == len(batch_ids):
        spec_files = input_files
        suffixes = input_suffixes
        use_input_files = True
    else:
        spec_files, suffixes = extract_fact_suffixes(args.specs_dir)
        use_input_files = False

    # If explicit batch IDs provided, map those IDs to input fact files deterministically
    if args.batch_ids:
        azure_log_dir = Path(args.azure_log_dir)
        batch_infos = load_batch_infos_by_ids(azure_log_dir, batch_ids)
        for b in batch_infos:
            b["upload_time"] = load_upload_time(azure_log_dir, b.get("input_file_id")) or b.get("meta_mtime", 0)
        # Restrict fact files to the same time window as these uploads (+/- 10 minutes)
        times = [b["upload_time"] for b in batch_infos if b.get("upload_time")]
        if times:
            lo = min(times) - 600
            hi = max(times) + 600
            fact_files = [p for p in input_files if lo <= p.stat().st_mtime <= hi]
            if not fact_files:
                fact_files = list(input_files)
        else:
            fact_files = list(input_files)
        mapped = map_batches_to_fact_files(batch_infos, fact_files)
        if not mapped:
            raise RuntimeError("Batch IDs provided but could not map to input fact files.")
        batch_ids = [b["id"] for b, _ in mapped]
        spec_files = [sf for _, sf in mapped]
        suffixes = []
        for p in spec_files:
            m = re.search(r"(_fact\d+)\.jsonl$", p.name)
            suffixes.append(m.group(1) if m else "")
        use_input_files = True

    if (not batch_ids) or (len(batch_ids) != len(suffixes)):
        # Attempt robust mapping via batch meta/upload times
        azure_log_dir = Path(args.azure_log_dir)
        batch_infos = load_batch_infos(azure_log_dir, args.since_minutes, args.max_batches)
        for b in batch_infos:
            b["upload_time"] = load_upload_time(azure_log_dir, b.get("input_file_id"))
        mapped = map_batches_to_fact_files(batch_infos, input_files if use_input_files else spec_files)
        if mapped:
            batch_ids = [b["id"] for b, _ in mapped]
            spec_files = [sf for _, sf in mapped]
            suffixes = []
            for p in spec_files:
                m = re.search(r"(_fact\d+)\.jsonl$", p.name)
                suffixes.append(m.group(1) if m else "")
            use_input_files = True if input_files else False
        else:
            if not batch_ids:
                raise RuntimeError("No batch IDs found and meta mapping failed.")
            min_len = min(len(batch_ids), len(suffixes))
            print(
                f"[WARN] Batch IDs ({len(batch_ids)}) != chunks ({len(suffixes)}); "
                f"proceeding with first {min_len} by order."
            )
            batch_ids = batch_ids[:min_len]
            spec_files = spec_files[:min_len]
            suffixes = suffixes[:min_len]

    azure_log_dir = Path(args.azure_log_dir)
    azure_log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uc = load_universe_context(args.universe_contexts_path)
    uc_id = uc.get("id") if isinstance(uc, dict) else None
    uc_is_true = uc.get("is_true") if isinstance(uc, dict) else None

    submit_manifest = load_submit_manifest(args.submit_manifest)
    if submit_manifest:
        manifest_facts = submit_manifest.get("facts", [])
        facts_with_chunks = [fact for fact in manifest_facts if fact.get("chunks")]
        if facts_with_chunks:
            for fact in facts_with_chunks:
                fact_suffix = fact.get("fact_suffix")
                if not fact_suffix:
                    continue
                synth_path = out_dir / f"{args.out_prefix}_{fact_suffix}.jsonl"
                if synth_path.exists() and not args.overwrite_merged:
                    print(f"[SKIP] merge exists: {synth_path.name}")
                    continue

                merged = 0
                with open(synth_path, "w") as out_f:
                    for chunk in fact.get("chunks", []):
                        batch_id = chunk.get("batch_id")
                        azure_input_path = chunk.get("azure_input_path")
                        if not batch_id or not azure_input_path:
                            continue

                        out_path = download_batch_output(
                            batch_id=batch_id,
                            azure_log_dir=azure_log_dir,
                            endpoint=endpoint,
                            api_version=api_version,
                            headers=headers,
                            overwrite_downloads=args.overwrite_downloads,
                        )
                        if out_path is None:
                            continue

                        specs = load_specs_from_input_file(Path(azure_input_path))
                        responses = load_responses(out_path)
                        for cid, text in responses.items():
                            if not cid:
                                continue
                            try:
                                idx = int(cid.split("_", 1)[0])
                            except Exception:
                                continue
                            if idx < 0 or idx >= len(specs):
                                continue

                            row = build_merged_row(
                                spec=specs[idx],
                                text=text,
                                uc_id=uc_id,
                                uc_is_true=uc_is_true,
                            )
                            row["custom_id"] = cid
                            out_f.write(json.dumps(row) + "\n")
                            merged += 1

                print(
                    f"[MERGE] {synth_path.name}: merged {merged} rows from {len(fact.get('chunks', []))} chunk(s)"
                )
            return

    for batch_id, spec_file, suffix in zip(batch_ids, spec_files, suffixes):
        out_path = download_batch_output(
            batch_id=batch_id,
            azure_log_dir=azure_log_dir,
            endpoint=endpoint,
            api_version=api_version,
            headers=headers,
            overwrite_downloads=args.overwrite_downloads,
        )
        if out_path is None:
            continue

        synth_path = out_dir / f"{args.out_prefix}{suffix}.jsonl"
        if synth_path.exists() and not args.overwrite_merged:
            print(f"[SKIP] merge exists: {synth_path.name}")
            continue

        specs = (
            load_specs_from_input_file(spec_file)
            if use_input_files
            else load_specs_from_spec_file(spec_file)
        )
        responses = load_responses(out_path)

        merged = 0
        with open(synth_path, "w") as out_f:
            for cid, text in responses.items():
                if not cid:
                    continue
                try:
                    idx = int(cid.split("_", 1)[0])
                except Exception:
                    continue
                if idx < 0 or idx >= len(specs):
                    continue

                row = build_merged_row(
                    spec=specs[idx],
                    text=text,
                    uc_id=uc_id,
                    uc_is_true=uc_is_true,
                )
                row["custom_id"] = cid

                out_f.write(json.dumps(row) + "\n")
                merged += 1

        print(f"[MERGE] {synth_path.name}: merged {merged} rows from {out_path.name}")


if __name__ == "__main__":
    main()
