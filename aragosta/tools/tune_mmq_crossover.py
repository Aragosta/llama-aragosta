#!/usr/bin/env python3
"""Derive the MMVQ -> MMQ crossover per quant type and op on the local GPU.

Runs `mmq-sweep` twice per (type, shape): GGML_CUDA_MMVQ_MAX=8 (MMVQ wherever the
built-in table allows) and =0 (always MMQ), then reports for every batch n which
kernel is faster and the largest n up to which MMVQ should be used.

    tune_mmq_crossover.py --bin build/bin/mmq-sweep --device CUDA1 --out results/crossover.json
"""
import argparse
import json
import os
import subprocess

# (label, extra args): a dense projection and the Qwen3.6-35B-A3B expert shapes (2048 hidden, 512 ffn, 256x8)
SHAPES = [
    ("dense_4096", ["--m", "4096", "--k", "4096"]),
    ("moe_up_512x2048", ["--m", "512", "--k", "2048", "--experts", "256", "--used", "8"]),
    ("moe_down_2048x512", ["--m", "2048", "--k", "512", "--experts", "256", "--used", "8"]),
]
TYPES = ["q4_0", "q4_K", "q5_K", "q6_K", "q8_0", "iq4_xs", "mxfp4"]


def sweep(binary, device, typ, shape_args, mmvq_max, nmax):
    env = dict(os.environ, GGML_CUDA_MMVQ_MAX=str(mmvq_max))
    cmd = [binary, "--type", typ, "--nmax", str(nmax), "--backend", device, *shape_args]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    return {r["n"]: r["us"] for r in map(json.loads, filter(None, p.stdout.splitlines()))}


def crossover(mmvq, mmq):
    """Largest n such that MMVQ is not slower than MMQ for every n' <= n (contiguous from 1)."""
    best = 0
    for n in sorted(mmvq):
        if mmvq[n] <= mmq[n]:
            best = n
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/bin/mmq-sweep")
    ap.add_argument("--device", default="CUDA0")
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="substring filter on shape label, e.g. moe")
    a = ap.parse_args()

    rows = []
    for typ in TYPES:
        for label, sargs in [x for x in SHAPES if a.only in x[0]]:
            v = sweep(a.bin, a.device, typ, sargs, a.nmax, a.nmax)  # dense clamps to 8 internally
            q = sweep(a.bin, a.device, typ, sargs, 0, a.nmax)
            if not v or not q:
                print(f"{typ:7s} {label:18s} unsupported")
                continue
            x = crossover(v, q)
            gain = {n: round(v[n] / q[n], 2) for n in sorted(v)}  # >1: MMQ faster
            rows.append({"type": typ, "shape": label, "mmvq_max": x, "mmvq_over_mmq": gain,
                         "mmvq_us": v, "mmq_us": q})
            print(f"{typ:7s} {label:18s} mmvq_max={x}  t_mmvq/t_mmq: " +
                  " ".join(f"{n}:{g}" for n, g in gain.items()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
