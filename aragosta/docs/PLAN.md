# Aragosta plan (rev 2, 2026-11-07)

## Critique of rev 1

1. **vLLM section was one bullet ("decode-first + prefill cap").** llama-server *already* adds decode
   tokens before prompt tokens (`update_slots`: `handle_last_sampled_token` runs first). The stall in
   #26022/#29175 is not ordering, it is *step latency*: one step carries up to `n_batch` prompt tokens,
   so every decoder waits for it. What actually needs porting from vLLM V1 is listed below (V1..V8).
2. **Bottleneck of the production model was ignored.** Qwen3.6-35B-A3B runs with `--n-cpu-moe 29`:
   most expert FFNs are on 8 CPU cores. Batched decode on CPU experts reads ~B distinct experts per
   layer, so cost grows ~linearly with users -> the measured flat c2=71 / c4=67 tok/s. GPU small-batch
   MMQ cannot fix that. The plan must (a) measure CPU-vs-GPU time per step, (b) minimise `n-cpu-moe`
   across both 3060s (layer split, quantised KV, smaller ctx per slot), (c) treat MMQ work as a win for
   GPU-resident layers and dense models only.
3. **MTP/spec under load.** Drafting helps c1 and hurts at c>=3 (verify batch = users x (1+draft)).
   vLLM disables speculation by batch size. Rev 1 never mentioned it; it is cheap and likely large.
4. **Marlin-style new kernel is the highest-risk/lowest-certainty item.** Ordered last; only if the
   crossover sweep shows a gap between MMVQ and MMQ at B=3..16 worth >10%.
5. **Order.** Scheduler + spec gating are server-only, fast to iterate and hit the agent workload
   directly. They go before KV-cache surgery (split-stream `seq_cp`) and kernels.
6. **No admission control / preemption story.** With unified KV and many slots the server can run out
   of cells; today `decode()` halves `n_batch` and retries — thrash, not scheduling.

## vLLM V1 ideas and how they map to llama-server

| # | vLLM V1 | llama-server today | Port |
|---|---------|--------------------|------|
| V1 | One token budget `max_num_batched_tokens`, `{req: n}` per step | prompt loop fills until `batch.size() < n_batch` | `--sched-budget` (tokens/step) separate from `n_batch`; per-slot allotment computed before the fill loop |
| V2 | `long_prefill_token_threshold`, not applied when request is alone | none | `--sched-prefill-chunk N`: cap per-slot prompt tokens per step **only while any slot is decoding** |
| V3 | adaptive threshold floored at fair share of budget | none | budget split round-robin among prefilling slots (fair share), remainder to oldest |
| V4 | FCFS / priority queue | prefill order = slot index | prefill order = task arrival time (FCFS); `priority` request field later |
| V5 | spec tokens counted in budget; disable spec by batch size | MTP drafts at any load | `--spec-max-active N`: skip drafting when > N slots generating |
| V6 | KV-block admission, recompute-preemption (LIFO) | fixed slots, halve-batch retry | phase 3: preempt newest slot to `--cache-ram`, requeue; replaces retry thrash |
| V7 | APC: hash-chained blocks, ref-counted, LRU free queue | LCP slot pick + RAM cache | phase 2 prefix index; unified KV cells are already ref-counted via seq sets |
| V8 | async scheduling, CUDA graphs per batch size, metrics | sync loop; graphs off for MMID sync path | measure CPU step overhead first; Prometheus: running/waiting/kv-usage/preempt/prefix-hit/ITL |
| V9 | DP replicas + prefix-aware router (llm-d) | n/a | `aragosta/router` only if DP beats layer split on 2x3060 |

## Order of execution

0. Build `aragosta` branch, baseline (dense 8B for iteration + production MoE profile).
1. Scheduler V1–V4 + spec gating V5 (server, opt-in flags, default = upstream behaviour). Gate: stall
   test (long prefill + decoders) max ITL bounded, aggregate tok/s within 5% of greedy.
2. Production-model profile: n-cpu-moe / split / KV-quant sweep across both GPUs.
3. MMVQ/MMQ crossover sweep (tool exists: `tools/mmq_sweep.cpp`, env `GGML_CUDA_MMVQ_MAX`), sm86 table.
4. Prefix cache for agents (#30664 design: cross-slot attach, pinned RAM prefix entries).
5. Split-stream partial `seq_cp` (finish #14363), KV-capacity preemption V6.
6. Kernel work only if 3 shows a gap.
