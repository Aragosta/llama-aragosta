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

## Hybrid AllReduce (`GGML_CUDA_ALLREDUCE=hybrid`, opt-in, NOT safe)

Reverted as default: two crashes in 3 runs, kernel log `NVRM: Xid 109, CTX SWITCH TIMEOUT` on GPU1 at the same times.
The internal kernels busy-wait on the peer GPU and the GPU cannot preempt them. NCCL-only ran every test clean.

Internal kernel AllReduce for messages under 1 MiB (decode, MTP verify), NCCL above (prefill). The internal
crash was in its copy-engine path (messages of 1 MiB and up), which hybrid never uses.

| Mode | pp2048 | tg128 |
|---|---|---|
| nccl | 843 | 34.7 |
| hybrid | 844 | 35.8 (+3.3%) |

Ceiling check (AllReduce skipped, wrong output, timing only): tg 37.9, pp 1199. So AllReduce costs ~6% of decode
but ~30% of prefill; prefill is limited by host-memory copy bandwidth (10 MB per AllReduce, 128 per ubatch).
Did not help prefill: `NCCL_SHM_USE_CUDA_MEMCPY`, `NCCL_MIN_NCHANNELS` 4/8, NTHREADS 512, internal kernel path for
large messages (801). CUDA graphs are already active (disabling them: tg 33.9 vs 35.8).
`-ub 2048` gives pp 864 (+2%) but runs out of VRAM at `-c 65536`.

## MTP depth and draft confidence (server, 1 / 2 users, tok/s)

| Setting | 1 user | 2 users |
|---|---|---|
| n-max 3 (kept) | 34.1 | 66.0 |
| n-max 2 | 33.3 | 57.6 |
| n-max 4 | 31.9 | failed |
| n-max 3, p-min 0.5 | 30.3 | 59.3 |

## 100k context and prefix anchors (VRAM per GPU)

| Build | recurrent state | loads at -c 100000 |
|---|---|---|
| anchors as full sequences (5 seqs x 4 rows) | 2993 MiB | no, out of memory |
| anchors attention-only (2 slots x 4 rows) | 1197 MiB | yes, 11.0 GB used |

Live check, NCCL, `-ub 512`: 34.4 tok/s at 1 user, 70.2 at 2, no errors. With speculation off the three anchor
agents give output identical to a cold run; with MTP on, greedy output can differ by float noise (batch shapes differ).

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

- Overlap prefill AllReduce with compute: needs two micro-batches in flight, which the single-ubatch llama_context cannot do. Prefill upper bound +42%.
- Real GPU P2P: `nvidia-smi topo -p2p r` says "Chipset not supported", BAR1 is 256 MiB.
- A rank-per-process executor.
