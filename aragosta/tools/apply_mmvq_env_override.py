#!/usr/bin/env python3
"""Source patch: GGML_CUDA_MMVQ_MAX overrides the dense MMVQ table instead of only lowering it.

Unset: per-arch tables decide. Set to N: dense quantized MUL_MAT uses MMVQ for n <= N, so N=8
reproduces upstream behaviour on archs without a table and N=0 forces MMQ (used by the tuner).
"""
import sys

P = "ggml/src/ggml-cuda/mmvq.cu"
OLD = """bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    return ne11 <= ggml_cuda_mmvq_env_max() && ggml_cuda_should_use_mmvq_table(type, cc, ne11);
}"""
NEW = """bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    static const bool env_set = getenv("GGML_CUDA_MMVQ_MAX") != nullptr;
    if (env_set && ggml_is_quantized(type)) {
        return ne11 <= ggml_cuda_mmvq_env_max(); // explicit override of the per-arch table
    }
    return ggml_cuda_should_use_mmvq_table(type, cc, ne11);
}"""
s = open(P).read()
if s.count(OLD) != 1:
    sys.exit("anchor not found exactly once")
open(P, "w").write(s.replace(OLD, NEW))
print("patched")
