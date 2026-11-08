# Aragosta

Serving changes to llama.cpp for 1-3 users running coding agents and sub-agents on 2x RTX 3060 12 GB
(sm_86, PCIe, no NVLink): long shared system/tool prefixes, short decode bursts, forks. Branch `aragosta`,
based on upstream `f7591a6a0` (2026-11-07). Everything is opt-in; without the flags the server is upstream.

## What is in it

| Area | Change | Flag | Where |
|------|--------|------|-------|
| Prefix sharing | A new task takes the longest shared prefix from any slot by sharing KV cells (`seq_cp`); hybrid models attach at checkpoints, with a checkpoint at the divergence point | `--slot-attach` (needs `--kv-unified`) | `server-context.cpp` `slot_attach()` |
| Prefix sharing | Persistent prefix anchors: KV of finished prompts is kept in extra sequences, independent of any slot. One anchor per agent/sub-agent prompt; a longer turn of the same prompt extends its anchor; LRU eviction | `--prefix-anchors N` (default 3) | `prefix_commit()`, `common_context_params_to_llama` |
| Scheduling | Prompt tokens per step are capped while any slot decodes (bounds decode stalls) | `--sched-prefill-chunk N` | `update_slots()` |
| Scheduling | Prefill in task-arrival order, not slot order | always | `prefill_order()` |
| Speculation | No new drafts while more than N slots generate (only useful for `-np` > 3) | `--spec-max-active N` | `update_slots()` |
| Speculation | A refused recurrent-state rollback ends the request instead of aborting the server | always | `server-context.cpp` |
| CUDA | sm86 MMVQ->MMQ crossover table, MoE MMVQ up to 16 tokens | `GGML_CUDA_MMVQ_MAX=0..32` | `ggml-cuda/mmvq.cu` |
| CUDA | `GGML_CUDA_ALLREDUCE=nccl` makes NCCL mandatory: startup fails instead of silently using the internal AllReduce | env | `ggml-cuda.cu` |

## Run it

```bash
aragosta/profiles/qwen38-27b-dense.sh              # Qwen3.8-27B, serves on 127.0.0.1:8080
PORT=8081 MODEL=/path/model.gguf aragosta/profiles/qwen38-27b-dense.sh --prefix-anchors 4
```

On the server it runs as the user service `llama-aragosta` (`systemctl --user status|restart llama-aragosta`).

## Build

```bash
cmake -S . -B build-nccl -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=ON \
  -DNCCL_INCLUDE_DIR=/home/snorkelaar/nccl/x/nvidia/nccl/include \
  -DNCCL_LIBRARY=/home/snorkelaar/nccl/x/nvidia/nccl/lib/libnccl.so.2
cmake --build build-nccl -j 8 --target llama-server llama-bench
ldd build-nccl/bin/libggml-cuda.so.0 | grep nccl     # must resolve to a path
```

`-DGGML_CUDA_NCCL=ON` alone is not enough: with the paths `NOTFOUND` it builds and silently falls back.

## Measure

- `bench/agent_load.py`, `bench/run_matrix.sh`: N agent sessions with a shared prefix; TTFT p50/p95, inter-token gap, stall, tok/s, cache hit %.
- `bench/anchors_check.py URL K`: K distinct long prompts, parents, erase slots, children must hit the anchors with identical greedy output (server needs `--slot-save-path`).
- `bench/attach_check.py`: attached prompt vs cold run vs same-slot reuse, greedy equality.
- `bench/prefix_ab.py`, `bench/nccl_ab.sh`: short A/B traces.
- `tools/mmq_sweep.cpp` + `tools/tune_mmq_crossover.py`: derive MMVQ/MMQ crossover tables for a GPU.

Numbers and decisions: [docs/RESULTS.md](docs/RESULTS.md). NCCL notes and rationale: [docs/NCCL.md](docs/NCCL.md).
