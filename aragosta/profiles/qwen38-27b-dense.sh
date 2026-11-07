#!/bin/bash
# Qwen3.8-27B Q4_K_M, dense, fully in VRAM on 2x RTX 3060 12GB.
# Measured (aragosta/results/d-*): vs default layer split without MTP,
#   1 user: 15.9 -> 34.6 tok/s, 2 users: 30.7 -> 64.9 tok/s aggregate.
# -sm tensor: both GPUs work on every token (layer split runs them one after another).
# MTP depth 3 beats 2 at 2 users (64.9 vs 59.9), ties at 1 user.
# --slot-attach: sessions share their common prompt prefix (agents).
# NCCL build (GGML_CUDA_NCCL, build-nccl): prefill +8% (790 -> 852 tok/s pp2048) but decode -3%
#   (36.9 -> 35.7 tok/s, 4 reps), so this profile uses the plain build; decode is the priority.
cd "$(dirname "$0")/../.."
exec ./build/bin/llama-server \
  -m /home/snorkelaar/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --jinja -ngl 99 -fa on -sm tensor \
  -c 65536 -np 2 --kv-unified -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 \
  --slot-attach --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port "${PORT:-8080}" "$@"
