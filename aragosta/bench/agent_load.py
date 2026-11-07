#!/usr/bin/env python3
"""Multi-session agentic load generator for llama-server (stdlib only).

Each session mimics a coding agent: a long shared system prompt (tool schemas +
instructions, identical across sessions), a short volatile tail (date / session
id), then several turns whose history grows with simulated tool output.

Reports per run: TTFT p50/p95, inter-token gap p95 and max stall, aggregate
decode tok/s, and prompt-cache efficiency (cache_n / prompt tokens, from the
server `timings` block).

    agent_load.py --url http://127.0.0.1:8081 --sessions 4 --turns 4 \
        --prefix-tokens 14000 --out results/run.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CHARS_PER_TOKEN = 3.6  # rough; only used to size synthetic text


def synth_tools(n_chars: int, seed: int = 7) -> str:
    """Deterministic fake tool-schema JSON of ~n_chars (identical for all sessions)."""
    rng = random.Random(seed)
    words = ["file", "path", "line", "query", "pattern", "edit", "diff", "branch", "commit",
             "shell", "timeout", "stdin", "range", "symbol", "index", "glob", "regex", "cwd"]
    out, i = [], 0
    while sum(map(len, out)) < n_chars:
        name = f"tool_{i}_{rng.choice(words)}"
        params = {w: {"type": "string", "description": " ".join(rng.choices(words, k=14))}
                  for w in rng.sample(words, 5)}
        out.append(json.dumps({"name": name, "description": " ".join(rng.choices(words, k=30)),
                               "parameters": {"type": "object", "properties": params}}))
        i += 1
    return "\n".join(out)


def synth_tool_output(n_chars: int, rng: random.Random) -> str:
    words = ["ok", "error", "warning", "def", "class", "return", "import", "None", "self", "0x1f"]
    return " ".join(rng.choices(words, k=max(1, int(n_chars / 5))))


def stream_chat(url: str, messages: list, max_tokens: int, extra: dict) -> dict:
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0.0,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}, **extra}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, text, timings = [], [], None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("timings"):
                timings = ev["timings"]
            ch = ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                stamps.append(time.perf_counter())
                text.append(ch[0]["delta"]["content"])
    return {"t0": t0, "stamps": stamps, "text": "".join(text), "timings": timings}


def run_session(sid: int, a, tools: str, go: threading.Barrier, results: list):
    rng = random.Random(1000 + sid)
    volatile = f"\nCurrent date: 2026-11-0{1 + sid % 9}. Session: {rng.getrandbits(48):012x}"
    msgs = [{"role": "system", "content": "You are a coding agent. Tools:\n" + tools + volatile}]
    go.wait()
    time.sleep(sid * a.stagger)  # stagger arrivals so new prefills land on running decoders
    for turn in range(a.turns):
        msgs.append({"role": "user", "content": f"Turn {turn}: inspect the repo and continue. "
                     + synth_tool_output(a.tool_tokens * CHARS_PER_TOKEN, rng)})
        r = stream_chat(a.url, msgs, a.max_tokens, {"cache_prompt": True})
        msgs.append({"role": "assistant", "content": r["text"] or "ok"})
        st = r["stamps"]
        gaps = [b - x for x, b in zip(st, st[1:])]
        tm = r["timings"] or {}
        results.append({"session": sid, "turn": turn, "ttft": (st[0] - r["t0"]) if st else None,
                        "n_tok": len(st), "gaps": gaps, "t_end": st[-1] if st else r["t0"],
                        "t0": r["t0"], "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                        "text": r["text"]})
        time.sleep(a.think_time)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def summarize(results: list) -> dict:
    ttft = [r["ttft"] for r in results if r["ttft"] is not None]
    gaps = [g for r in results for g in r["gaps"]]
    toks = sum(r["n_tok"] for r in results)
    wall = max(r["t_end"] for r in results) - min(r["t0"] for r in results)
    pn = sum(r["prompt_n"] or 0 for r in results)
    cn = sum(r["cache_n"] or 0 for r in results)
    return {"requests": len(results), "wall_s": round(wall, 2), "agg_tok_s": round(toks / wall, 2),
            "ttft_p50_s": round(statistics.median(ttft), 3), "ttft_p95_s": round(pct(ttft, 95), 3),
            "gap_p95_ms": round(pct(gaps, 95) * 1e3, 1), "max_stall_s": round(max(gaps, default=0), 3),
            "prompt_tokens": pn, "cache_hit_tokens": cn,
            "cache_hit_pct": round(100 * cn / (pn + cn), 1) if pn + cn else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--prefix-tokens", type=int, default=14000)
    p.add_argument("--tool-tokens", type=int, default=400, help="simulated tool output per turn")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--think-time", type=float, default=0.0)
    p.add_argument("--stagger", type=float, default=0.0, help="session i starts i*S seconds late")
    p.add_argument("--out", default="")
    p.add_argument("--label", default="")
    a = p.parse_args()

    tools = synth_tools(int(a.prefix_tokens * CHARS_PER_TOKEN))
    # warm-up on a tiny unrelated prompt so the first run does not pay lazy init
    stream_chat(a.url, [{"role": "user", "content": "hi"}], 4, {"cache_prompt": False})
    results, go = [], threading.Barrier(a.sessions)
    with ThreadPoolExecutor(a.sessions) as ex:
        for s in range(a.sessions):
            ex.submit(run_session, s, a, tools, go, results)
    summ = summarize(results)
    summ.update(label=a.label, sessions=a.sessions, stagger=a.stagger, max_tokens=a.max_tokens, turns=a.turns, prefix_tokens=a.prefix_tokens)
    print(json.dumps(summ))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summ, "requests": results}, f)


if __name__ == "__main__":
    main()
