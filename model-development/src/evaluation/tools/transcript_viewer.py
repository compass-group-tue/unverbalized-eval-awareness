#!/usr/bin/env python3
"""Serve a lightweight local viewer for saved generation-transcript JSONLs.

The viewer intentionally accepts the canonical transcript fields used by the
final experiments and the older Stage-1/full-validation exports: ``id``,
``input_messages``, ``reasoning``, and ``response``. It does not alter data;
all rows are read into memory and served locally on loopback only by default.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Transcript viewer</title>
<style>
:root { color-scheme: light; --ink:#172033; --muted:#516079; --line:#d9e0ea; --panel:#fff; --page:#f5f7fb; --accent:#2563eb; }
* { box-sizing:border-box; } body { margin:0; background:var(--page); color:var(--ink); font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }
header { position:sticky; top:0; z-index:2; display:flex; align-items:center; gap:10px; padding:12px 20px; background:#fff; border-bottom:1px solid var(--line); }
h1 { margin:0 auto 0 0; font-size:18px; } button,select { padding:6px 9px; border:1px solid #b8c4d5; border-radius:6px; background:#fff; color:var(--ink); } button { cursor:pointer; } button:hover { background:#eef4ff; }
#counter { min-width:85px; text-align:center; color:var(--muted); } main { max-width:1400px; margin:20px auto; padding:0 20px 42px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:13px 15px; margin:12px 0; } .meta { display:flex; gap:9px 22px; flex-wrap:wrap; } .meta span { color:var(--muted); } .meta b { color:var(--ink); }
h2 { margin:0 0 8px; font-size:15px; } .message { border-left:4px solid #94a3b8; padding:8px 11px; margin:8px 0; background:#f8fafc; border-radius:0 6px 6px 0; } .message.system { border-color:#7c3aed; } .message.user { border-color:#2563eb; } .message.assistant { border-color:#059669; }
.role { font-size:11px; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px; } pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
.columns { display:grid; grid-template-columns:1fr 1fr; gap:14px; } .reasoning { border-top:4px solid #8b5cf6; } .response { border-top:4px solid #059669; } .judge { border-top:4px solid #d97706; } .verdict { font-weight:800; } .verdict.yes { color:#b91c1c; } .verdict.no { color:#047857; } .verdict.unavailable { color:var(--muted); } details { margin-top:12px; } summary { cursor:pointer; color:var(--muted); } .empty { color:var(--muted); font-style:italic; } @media(max-width:850px){ .columns{grid-template-columns:1fr;} }
</style></head><body><header><h1 id="title">Transcript viewer</h1><select id="source"></select><button id="prev">← Prev</button><span id="counter"></span><button id="next">Next →</button></header><main id="app"></main>
<script>
let allRows=[], rows=[], index=0; const $=x=>document.querySelector(x);
function add(parent, tag, text, cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;parent.appendChild(el);return el;}
function card(parent, title, cls){const el=add(parent,'section',undefined,`card ${cls||''}`);add(el,'h2',title);return el;}
function messages(parent, values){const box=card(parent,'Conversation prefix');if(!values.length){add(box,'div','No saved input messages','empty');return;}for(const m of values){const item=add(box,'div',undefined,`message ${String(m.role||'').toLowerCase()}`);add(item,'div',String(m.role||'message'),'role');add(item,'pre',String(m.content||''));}}
function judgePanel(parent,r){const box=card(parent,'Judge result','judge');const outcome=String(r.judge_outcome||'unavailable');add(box,'div',`Verdict: ${outcome}`,`verdict ${outcome.toLowerCase()}`);const meta=add(box,'div',undefined,'meta');for(const [k,v] of [['protocol',r.judge_protocol||'—'],['status',r.judge_status||'—'],['finish reason',r.judge_finish_reason||'—'],['evidence status',r.judge_evidence_status||'—']]){const item=add(meta,'div');add(item,'span',`${k}: `);add(item,'b',String(v));}add(box,'h2','Evidence quote');add(box,'pre',r.judge_evidence_quote?String(r.judge_evidence_quote):'No evidence quote recorded.',r.judge_evidence_quote?'':'empty');if(r.judge_raw_response){const details=add(box,'details');add(details,'summary','Raw judge response');add(details,'pre',String(r.judge_raw_response));}}
function render(){const app=$('#app');app.innerHTML='';if(!rows.length){add(app,'p','No rows match this filter.','empty');return;}index=(index+rows.length)%rows.length;const r=rows[index];const hasResponse=Object.prototype.hasOwnProperty.call(r,'response');$('#counter').textContent=`${index+1} / ${rows.length}`;const meta=card(app,'Sample metadata','meta');for(const [k,v] of [['ID',r.record_id||r.id||'—'],['model',r.model_label||r._viewer_source],['setting',r.system_prompt_condition||r.source_type||r.metadata?.source_type||'—'],['dataset',r.dataset||r.source_dataset||r.metadata?.source_dataset||'—'],['assistant turn',r.assistant_turn_index??'—'],['prompt source',r.input_messages_source||'—'],['marker response',hasResponse?String(String(r.response||'').includes('§')):'not included'],['marker reasoning',String(String(r.reasoning||'').includes('§'))]]){const item=add(meta,'div');add(item,'span',`${k}: `);add(item,'b',String(v));}messages(app,Array.isArray(r.input_messages)?r.input_messages:[]);if(window.viewerShowJudge)judgePanel(app,r);const cols=add(app,'section',undefined,'columns');const reason=card(cols,'Reasoning','reasoning');add(reason,'pre',r.reasoning?String(r.reasoning):'No saved reasoning',''+(!r.reasoning?'empty':''));if(hasResponse){const response=card(cols,'Visible answer','response');add(response,'pre',r.response?String(r.response):'Empty visible answer',''+(!r.response?'empty':''));}}
function setRows(){const source=$('#source').value;rows=source==='all'?allRows:allRows.filter(r=>r._viewer_source===source);index=0;render();}
fetch('/api/rows').then(r=>r.json()).then(data=>{allRows=data.rows;window.viewerShowJudge=Boolean(data.show_judge);$('#title').textContent=data.title;for(const source of ['all',...new Set(allRows.map(r=>r._viewer_source))]){const opt=add($('#source'),'option',source);opt.value=source;}setRows();});$('#source').onchange=setRows;$('#prev').onclick=()=>{index--;render()};$('#next').onclick=()=>{index++;render()};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')$('#prev').click();if(e.key==='ArrowRight')$('#next').click();});
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, nargs="+", required=True, help="One or more saved transcript JSONLs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--title", default="Generation transcript viewer")
    parser.add_argument(
        "--show-judge",
        action="store_true",
        help="Display judge verdict, evidence quote, and raw judge response when present.",
    )
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Transcript JSONL not found: {path}")
        for line_number, raw in enumerate(path.open(encoding="utf-8"), 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            item["_viewer_source"] = path.stem
            rows.append(item)
    if not rows:
        raise ValueError("No transcript rows loaded")
    return rows


def main() -> int:
    args = parse_args()
    rows = load_rows(args.jsonl)
    payload = json.dumps(
        {"title": args.title, "rows": rows, "show_judge": args.show_judge},
        ensure_ascii=False,
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                body, content_type = HTML.encode("utf-8"), "text/html; charset=utf-8"
            elif path == "/api/rows":
                body, content_type = payload, "application/json; charset=utf-8"
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[transcript-viewer] rows={len(rows)} url={url}", flush=True)
    if args.open_browser:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[transcript-viewer] stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
