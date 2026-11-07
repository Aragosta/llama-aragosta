#!/usr/bin/env python3
"""Source patch: MMVQ for MUL_MAT_ID beyond 8 tokens.

The multi-token MoE MMVQ kernel (mul_mat_vec_q_moe) runs one warp per token and shares no weights
between tokens of a block, so the 8-token cap is a block-shape limit, not an algorithmic one. We tile
tokens over gridDim.z (8 per block) and let arch tables allow up to MMVQ_MMID_MAX_BATCH_SIZE tokens.
On an RTX 3060, MMVQ still beats MMQ by ~30% at 8 tokens for Qwen3-MoE expert shapes, and
multi-user decode with MTP verify (users x (1 + n_draft)) lands right in 9..16.

GGML_CUDA_MMVQ_MAX now accepts 0..MMVQ_MMID_MAX_BATCH_SIZE; dense MUL_MAT clamps it to
MMVQ_MAX_BATCH_SIZE.
"""
import sys


def rep(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: anchor matched {n}x (want {count}): {old[:80]!r}")
    open(path, "w").write(s.replace(old, new))


H = "ggml/src/ggml-cuda/mmvq.cuh"
C = "ggml/src/ggml-cuda/mmvq.cu"
G = "ggml/src/ggml-cuda/ggml-cuda.cu"

rep(H, "#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.\n",
       "#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.\n"
       "#define MMVQ_MMID_MAX_BATCH_SIZE 32 // aragosta: max. tokens for the MUL_MAT_ID MMVQ kernel (tiled by 8 over gridDim.z)\n")

# --- env parsing: allow up to the MMID cap
rep(C, "        if (*end != '\\0' || v < 0 || v > MMVQ_MAX_BATCH_SIZE) {\n"
       "            GGML_LOG_WARN(\"%s: ignoring invalid GGML_CUDA_MMVQ_MAX='%s' (expected 0..%d)\\n\", __func__, s, MMVQ_MAX_BATCH_SIZE);\n"
       "            return (int) MMVQ_MAX_BATCH_SIZE;\n",
       "        if (*end != '\\0' || v < 0 || v > MMVQ_MMID_MAX_BATCH_SIZE) {\n"
       "            GGML_LOG_WARN(\"%s: ignoring invalid GGML_CUDA_MMVQ_MAX='%s' (expected 0..%d)\\n\", __func__, s, MMVQ_MMID_MAX_BATCH_SIZE);\n"
       "            return -1;\n")
rep(C, """        if (s == nullptr || *s == '\\0') {
            return (int) MMVQ_MAX_BATCH_SIZE;
        }""", """        if (s == nullptr || *s == '\\0') {
            return -1; // unset
        }""")

# --- MMID host cap: table, extended on tuned archs, env override
rep(C, """int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    return std::min(get_mmvq_mmid_max_batch_table(type, cc), ggml_cuda_mmvq_env_max());
}""", """int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    const int table = get_mmvq_mmid_max_batch_table(type, cc);
    const int env   = ggml_cuda_mmvq_env_max();
    if (table < MMVQ_MAX_BATCH_SIZE) {
        // the device kernel is compiled for <= table tokens per block, so the token tiling below does not apply
        return env >= 0 ? std::min(table, env) : table;
    }
    if (env >= 0) {
        return env;
    }
    // aragosta: tuned on RTX 3060 (GA106) with tune_mmq_crossover.py --nmax 32 on Qwen3-MoE expert shapes
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_AMPERE_GA10X) {
        return ggml_cuda_mmvq_mmid_ext_sm86(type);
    }
    return table;
}""")

rep(C, "static int get_mmvq_mmid_max_batch_table(ggml_type type, int cc);\n",
       """static int get_mmvq_mmid_max_batch_table(ggml_type type, int cc);

static int ggml_cuda_mmvq_mmid_ext_sm86(ggml_type type) {
    switch (type) {
        default: return MMVQ_MAX_BATCH_SIZE; // filled in from the sweep
    }
}
""")

# --- dense path: unset env -> table; set -> clamp to the dense cap
rep(C, """    static const bool env_set = getenv("GGML_CUDA_MMVQ_MAX") != nullptr;
    if (env_set && ggml_is_quantized(type)) {
        return ne11 <= ggml_cuda_mmvq_env_max(); // explicit override of the per-arch table
    }""", """    const int env = ggml_cuda_mmvq_env_max();
    if (env >= 0 && ggml_is_quantized(type)) {
        // explicit override of the per-arch table, clamped to the dense kernel's compiled limit
        return ne11 <= std::min(env, (int) MMVQ_MAX_BATCH_SIZE);
    }""")

# --- kernel: token index includes the z tile
rep(C, "    const uint32_t token_idx   = threadIdx.y;\n",
       "    const uint32_t token_idx   = blockIdx.z*blockDim.y + threadIdx.y; // aragosta: tokens tiled over gridDim.z\n")

# --- launcher
rep(C, """    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);""", """    // aragosta: at most MMVQ_MAX_BATCH_SIZE tokens (warps) per block, matching __launch_bounds__;
    // larger batches are tiled over gridDim.z. Each warp is independent, so this is exact.
    const uint32_t tokens_per_block = std::min<uint32_t>(ncols_dst, MMVQ_MAX_BATCH_SIZE);
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr),
                          (ncols_dst + tokens_per_block - 1) / tokens_per_block);
    const dim3 block_dims(warp_size, tokens_per_block);""")

rep(C, "    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);\n",
       "    GGML_ASSERT(ncols_dst <= (ids ? MMVQ_MMID_MAX_BATCH_SIZE : MMVQ_MAX_BATCH_SIZE));\n")
rep(C, "    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);\n",
       "    GGML_ASSERT(!ids || ne12 <= MMVQ_MMID_MAX_BATCH_SIZE);\n")

# --- dispatch sites in ggml-cuda.cu
rep(G, """    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {""", """    if (ggml_is_quantized(src0->type)) {
        if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
            return false;
        }
    } else if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (GGML_CUDA_CC_IS_AMD(cc)) {""")
rep(G, """        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }""", """        if (ggml_is_quantized(src0->type)) {
            if (ne2 <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                return;
            }
        } else if (ne2 <= MMVQ_MAX_BATCH_SIZE && GGML_CUDA_CC_IS_AMD(cc)) {
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
            return;
        }""")
print("patched")
