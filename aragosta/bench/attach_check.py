#!/usr/bin/env python3
"""Correctness check for --slot-attach: greedy output of a prompt that attaches to another slot's
prefix must match the same prompt computed cold, as closely as ordinary same-slot cache reuse does.

Needs a server started with -np >= 3 --kv-unified [--slot-attach]. Uses /completion with id_slot so
the donor and the receiver are known slots, and natural text (a docs file) so that greedy decoding is
not dominated by near-ties.

    attach_check.py --url http://127.0.0.1:8081 --context docs/build.md
"""
import argparse
import json
import urllib.request


def complete(url, prompt, slot, n, cache=True):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "id_slot": slot,
            "cache_prompt": cache, "return_tokens": True}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["tokens"], d.get("timings", {})


def erase(url, slot):
    req = urllib.request.Request(f"{url}/slots/{slot}?action=erase", b"", {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()


def agree(a, b):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return n, sum(x == y for x, y in zip(a, b)) / max(1, min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--context", required=True, help="text file used as the shared prefix")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ctx = open(a.context).read()
    sys_a = f"Reference documentation:\n{ctx}\n\nCurrent date: 2026-11-03.\n"
    sys_b = f"Reference documentation:\n{ctx}\n\nCurrent date: 2026-11-07.\n"
    q_a = "Question: summarise the build options for CUDA in five bullet points.\nAnswer:"
    q_b = "Question: which CMake flag enables the Vulkan backend, and what does it need installed?\nAnswer:"
    pb = sys_b + q_b

    for s in range(3):
        erase(a.url, s)
    cold, t_cold = complete(a.url, pb, 2, a.n, cache=False)       # reference: computed from scratch

    erase(a.url, 2)
    complete(a.url, sys_a + q_a, 0, a.n)                           # donor: slot 0 holds the shared prefix
    att, t_att = complete(a.url, pb, 1, a.n)                       # receiver: empty slot 1 -> attach (if on)
    again, t_again = complete(a.url, pb, 1, a.n)                   # noise floor: same-slot cache reuse

    res = {
        "prompt_tokens_cold": t_cold.get("prompt_n"), "prompt_tokens_attach": t_att.get("prompt_n"),
        "cache_tokens_attach": t_att.get("cache_n"),
        "ttft_ms_cold": t_cold.get("prompt_ms"), "ttft_ms_attach": t_att.get("prompt_ms"),
        "attach_vs_cold": agree(att, cold), "reuse_vs_cold": agree(again, cold),
    }
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
