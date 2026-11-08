#!/bin/bash
# Qwen3.8-27B Q4_K_M, dense, fully in VRAM on 2x RTX 3060 12GB. This is the live profile.
# -sm tensor + native NCCL AllReduce (fails fast without NCCL; the internal AllReduce kernels spin-wait and
#   caused Xid 109 CTX SWITCH TIMEOUT crashes). 844 pp2048 / 34.7 tg128 tok/s single stream.
#   pipeline crashed with CUDA launch failures). 844 pp2048 / 34.7 tg128 tok/s single stream.
#   NCCL env knobs (PROTO/ALGO/NTHREADS/BUFFSIZE) were swept: none beat the defaults. The two
#   3060s have no GPU P2P, so NCCL moves data through host shared memory.
# MTP depth 3: 1 user ~37, 2 users ~65-69 tok/s aggregate.
# --slot-attach + --prefix-anchors 3: agents and sub-agents share long system prompts; 3 distinct
#   prompts stay resident (warm TTFT 3.8 s -> 0.4 s at 2.4k tokens) with greedy-identical output.
# --sched-prefill-chunk 512: caps prompt tokens per step while someone decodes
#   (max decode stall 2.8 s -> 0.9 s with a long prefill arriving mid-decode, -1.7% aggregate tok/s).
# 100k context fits at 11.1 GB/GPU: prefix anchors hold attention KV only (recurrent state of an anchor lives in
#   host checkpoints), so they cost no recurrent-state VRAM. -ub 1024 needs -c <= 65536.
# Override: MODEL=/path/to.gguf PORT=8080 NCCL_LIB=/dir/with/libnccl.so.2, extra args are appended.
cd "$(dirname "$0")/../.."
MODEL=${MODEL:-/home/snorkelaar/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf}
export LD_LIBRARY_PATH="${NCCL_LIB:-/home/snorkelaar/nccl/x/nvidia/nccl/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_ALLREDUCE=nccl
if ! ldd ./build-nccl/bin/libggml-cuda.so.0 | grep -q 'libnccl.so.2 => /'; then
  echo 'NCCL-linked CUDA backend required: see aragosta/README.md (Build)' >&2
  exit 1
fi
exec ./build-nccl/bin/llama-server \
  -m "$MODEL" --alias qwen3.8-27b \
  --jinja --reasoning-format deepseek --no-mmproj -ngl 99 -fa on -fit off -sm tensor -ts 1,1 -t 4 \
  -c 100000 -np 2 --kv-unified -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 \
  --slot-attach --prefix-anchors 3 --sched-prefill-chunk 512 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
