#!/usr/bin/env python3
"""Sub-agent prefix test: K distinct long system prompts. Parents run once, slots are erased,
then a child of every prompt must hit the persistent prefix cache and give the same greedy output."""
import json
import sys
import urllib.request

from agent_load import CHARS_PER_TOKEN, stream_chat, synth_tools

url, k = sys.argv[1], int(sys.argv[2])
ptok = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
systems = [f"You are agent {i}. Tools:\n" + synth_tools(int(ptok * CHARS_PER_TOKEN), seed=11 + i) for i in range(k)]


def ask(system, q):
    r = stream_chat(url, [{"role": "system", "content": system}, {"role": "user", "content": q}], 24, {"cache_prompt": True})
    t = r["timings"] or {}
    return t.get("cache_n"), t.get("prompt_n"), r["stamps"][0] - r["t0"], "".join(r["text"])


def erase():
    for sid in range(2):
        urllib.request.urlopen(urllib.request.Request(f"{url}/slots/{sid}?action=erase", data=b"", method="POST"), timeout=30).read()


cold = [ask(s, "Say hi.") for s in systems]
erase()
warm = [ask(s, "Say hi.") for s in systems]
ok = True
for i, (c, w) in enumerate(zip(cold, warm)):
    hit = (w[0] or 0) > 0.9 * ((w[0] or 0) + (w[1] or 0))
    same = c[3] == w[3]
    ok &= hit and same
    if not same: print(repr(c[3]), repr(w[3]))
    print(json.dumps({"agent": i, "cold_ttft": round(c[2], 2), "warm_ttft": round(w[2], 2),
                      "warm_cache_n": w[0], "warm_prompt_n": w[1], "hit": hit, "same_output": same}))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
