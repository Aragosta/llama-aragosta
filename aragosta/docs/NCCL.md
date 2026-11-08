# NCCL tensor parallel production path

## Decision

Keep Aragosta's existing tensor-sharded graph and CUDA NCCL AllReduce. Select NCCL explicitly for the production dense profile and fail startup if it is unavailable. This is the smallest architecture change that removes the observed crash path without replacing ggml's scheduler or model loader.

## Evidence and vLLM notes

- The server has two RTX 3060 12 GB GPUs with a PCIe host-bridge path and no NVLink. The crash log ends at `ggml_cuda_ar_acquire_slot` in the internal AllReduce pipeline after an unspecified CUDA launch failure.
- `build/CMakeCache.txt` says `GGML_CUDA_NCCL=ON`, but `NCCL_INCLUDE_DIR` and `NCCL_LIBRARY` are `NOTFOUND`. That binary reports "NCCL not compiled in" and silently falls back to internal AllReduce.
- `build-nccl/` links `libnccl.so.2`. The prior 1- and 2-session A/B showed similar throughput for NCCL and internal AllReduce; it did not establish long-run stability.
- vLLM uses tensor-parallel ranks and collective communication, with NCCL as a supported path. It validates worker placement against available GPUs, and offers NCCL as a fallback to its custom AllReduce. The transferable idea here is a verified collective backend with an explicit selection and observable initialization. vLLM's worker-process topology and scheduler do not map directly to ggml's in-process meta backend.

Sources: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/ and https://docs.vllm.ai/en/stable/api/vllm/config/parallel/ and https://docs.vllm.ai/en/stable/api/vllm/distributed/parallel_state/

## Runtime contract

1. Build with CUDA and NCCL, supplying the actual include and library paths. Check `libggml-cuda.so.0` links `libnccl.so.2`; the CMake option alone does not prove that.

   `cmake -S . -B build-nccl -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=/home/snorkelaar/nccl/x/nvidia/nccl/include -DNCCL_LIBRARY=/home/snorkelaar/nccl/x/nvidia/nccl/lib/libnccl.so.2`

   `cmake --build build-nccl -j 4 --target llama-server`

2. The production profile sets `GGML_CUDA_ALLREDUCE=nccl`, checks the linked library, and launches `build-nccl`. `NCCL_DEBUG=INFO` exposes communicator initialization in the server log. An explicitly requested NCCL backend aborts startup if initialization fails, instead of entering the crash-prone internal pipeline.
3. Keep the existing `-sm tensor` graph partitioning, BF16 conversion for large reductions, and per-device streams. Keep auto fallback behavior for other profiles and platforms.
4. Validate with a long prompt, decode, multiple concurrent sessions, and model output parity. Record crashes, request errors, token rate, TTFT, and inter-token stalls. Run a wider overnight soak before calling this fully production proven.

## Follow-up architecture if this gate fails

If native NCCL still fails under sustained load, isolate each GPU in a worker process using ggml's existing `comm_init_rank` interface, add a coordinator-owned restart policy, and keep one request queue. That is a larger change with new lifecycle and recovery behavior and should be justified by a reproduced NCCL failure first. Do not add a second tensor planner or a custom collective kernel now.

## Rollback

The previous binary and profile are retained. To roll back this deployment, launch the old `build/bin/llama-server` profile command without `GGML_CUDA_ALLREDUCE=nccl`. This restores the old crash risk and is for emergency recovery only.
