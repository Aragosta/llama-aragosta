#!/bin/bash
cd ~/llama-aragosta
export LD_LIBRARY_PATH=$HOME/nccl/x/nvidia/nccl/lib
M=/home/snorkelaar/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf
export SESSIONS="1 2" PREFIX=4000 TURNS=2 LOAD_ARGS="--max-tokens 256"
C="-c 65536 -np 2 --kv-unified -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 --slot-attach -sm tensor --spec-type draft-mtp --spec-draft-n-max 3"
aragosta/bench/run_matrix.sh n-plain ./build/bin/llama-server $M -- $C
aragosta/bench/run_matrix.sh n-nccl ./build-nccl/bin/llama-server $M -- $C
echo DONE
