# Results (2x RTX 3060 12 GB, Qwen3.8-27B Q4_K_M, `-sm tensor`)

Raw per-run JSON/logs were removed from the repo; these are the numbers that drove decisions.

## Single stream, llama-bench (ub 1024, q8_0 KV, 2026-11-08)

| AllReduce | pp2048 tok/s | tg128 tok/s |
|---|---|---|
| NCCL | 844 | 34.7 |
| internal | 784 | 35.9 |

NCCL env sweep (`NCCL_PROTO` LL/LL128/Simple, `NCCL_ALGO=Tree`, `NCCL_NTHREADS=64`, `NCCL_MAX_NCHANNELS=1`,
`NCCL_BUFFSIZE`, `NCCL_P2P_*`): nothing beat the defaults (LL: pp -22%; Simple/Tree: tg -6%).
`NCCL_DEBUG=INFO` shows channels run over `SHM/direct`: the 3060s expose no GPU P2P, so every
AllReduce goes through host shared memory. Decode is ~29 ms/token against ~21 ms for the weights alone,
so ~8 ms/token is communication and launch latency. Removing it needs fewer or overlapped AllReduces
inside `ggml-backend-meta.cpp`, not NCCL settings. `nsys` cannot import reports on this host (importer missing).

## Server, MTP3 (earlier A/B)

| Setup | 1 user | 2 users |
|---|---|---|
| layer split, no MTP | 15.9 | 30.7 |
| tensor split + MTP3 | 34.6-37 | 64.9-69 |

MTP3 vs MTP2 at 2 users: 64.9 vs 59.9 tok/s. NCCL vs internal AllReduce at 2 users: 69.4 vs 68.9 (tie).

## Prefix anchors (`bench/anchors_check.py`, 2.4k-token prompts)

| Agents / anchors | Cold TTFT | Warm TTFT | Cache hit | Output vs cold |
|---|---|---|---|---|
| 3 / 3 | 3.8 s | 0.3-0.5 s | 2403 of 2406 tokens | identical |
| 4 / 3 | 3.7 s | 3.7 s for the evicted one | LRU eviction | identical |

## Decode stalls while a long prompt arrives (2 sessions, 12k-token prompts, 4 s stagger)

| `--sched-prefill-chunk` | max stall | agg tok/s | TTFT p95 |
|---|---|---|---|
| off | 2.78 s | 14.71 | 23.2 s |
| 1024 | 1.47 s | 14.67 | 23.3 s |
| 512 | 0.88 s | 14.46 | 23.9 s |

## Not done (needs a profiler or a larger change)

- Overlap AllReduce with compute / fewer AllReduces per layer in `ggml-backend-meta.cpp`.
- CUDA-graph capture of the NCCL decode step (NCCL supports capture, but several GPUs in one process can deadlock).
- A rank-per-process executor.
