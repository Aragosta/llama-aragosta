#!/usr/bin/env python3
"""Source patch: sm86 (GA10x) MMVQ -> MMQ crossover table in ggml-cuda/mmvq.cu.

Thresholds measured on an RTX 3060 12GB with tune_mmq_crossover.py (results/crossover_3060.json).
"""
import sys

P = "ggml/src/ggml-cuda/mmvq.cu"
ANCHOR = "    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {\n"
BLOCK = """    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_AMPERE_GA10X) {
        // aragosta: tuned on RTX 3060 12GB (GA106, 360 GB/s), 4096x4096 weights, with
        // aragosta/tools/tune_mmq_crossover.py. Low bandwidth relative to int8 MMA throughput lets
        // MMQ win early on K-quants (MMVQ is 46% slower than MMQ for q4_K at n=8).
        switch (type) {
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 4;
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_MXFP4:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
"""

s = open(P).read()
if "cc == GGML_CUDA_CC_AMPERE_GA10X) {\n        // aragosta" in s:
    sys.exit("already applied")
if s.count(ANCHOR) != 1:
    sys.exit("anchor not found exactly once")
open(P, "w").write(s.replace(ANCHOR, BLOCK + ANCHOR))
print("patched")
