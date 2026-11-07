# Aragosta

Serving-oriented changes to llama.cpp for multi-user, agentic workloads: many sessions sharing long
tool/system prefixes, short decode bursts, tool-call round trips. Branch `aragosta`, based on upstream
`f7591a6a0` (2026-11-07). Everything is opt-in; with no new flags the server behaves as upstream.

Target hardware for the numbers here: 2x RTX 3060 12 GB (sm_86, PCIe, no NVLink), 8-core CPU, 30 GB RAM.

## What is in it

| Area | Change | Flag / knob | Where |
|------|--------|-------------|-------|
| Prefix sharing | Cross-slot attach: a new task takes the longest shared prefix from any slot by sharing its KV cells (`seq_cp`); hybrid models attach at checkpoints. After [#30664](https://github.com/ggml-org/llama.cpp/discussions/30664). | `--slot-attach` (needs `--kv-unified`) | `tools/server/server-context.cpp` `slot_attach()` |
| Prefix sharing | Divergence checkpoints: hybrid prompts checkpoint where they leave the best cached prompt, so later sessions resume exactly there | on with `--slot-attach` | same, `n_ckpt_divergence` |
| Scheduling | Prefill token budget while any slot decodes (vLLM `long_prefill_token_threshold`); lifted when idle | `--sched-prefill-chunk N` | `update_slots()` |
| Scheduling | Prompt processing in task-arrival (FCFS) order instead of slot index | always | `prefill_order()` |
| Speculation | No new MTP/draft rounds while more than N slots generate (vLLM disables spec by batch size) | `--spec-max-active N` | `update_slots()` |
| CUDA | sm86 (GA10x) MMVQ->MMQ crossover table for dense matmuls | automatic on sm86 | `ggml-cuda/mmvq.cu` |
| CUDA | MoE MMVQ kernel past 8 tokens (tokens tiled over `gridDim.z`), sm86 table up to 16 | automatic on sm86 | `ggml-cuda/mmvq.cu` |
| CUDA | Runtime override of both tables | `GGML_CUDA_MMVQ_MAX=0..32` | `ggml-cuda/mmvq.cu` |

## Tools

- `bench/agent_load.py` — N concurrent agent sessions, shared tool prefix + volatile tail, growing history.
  Reports TTFT p50/p95, inter-token gap p95, max decode stall, aggregate tok/s, prompt-cache hit %.
- `bench/run_matrix.sh` — start a server profile, run `agent_load.py` at several concurrencies, save JSON.
- `bench/attach_check.py` — greedy-output equality of an attached prompt vs a cold run vs same-slot reuse.
- `tools/mmq_sweep.cpp` (`mmq-sweep`) + `tools/tune_mmq_crossover.py` — time MMVQ vs MMQ per quant type and
  shape, derive crossover tables for the local GPU.

## Reproduce

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j --target llama-server mmq-sweep test-backend-ops
python3 aragosta/tools/tune_mmq_crossover.py --device CUDA0 --nmax 32 --out aragosta/results/crossover.json
GGML_CUDA_MMVQ_MAX=32 ./build/bin/test-backend-ops -o MUL_MAT_ID -b CUDA0
SESSIONS="1 2 4" aragosta/bench/run_matrix.sh label ./build/bin/llama-server model.gguf -- -np 4 --kv-unified --slot-attach
```

Results: [docs/RESULTS.md](docs/RESULTS.md). Plan, critique and vLLM mapping: [docs/PLAN.md](docs/PLAN.md).
