// mmq-sweep: time quantized MUL_MAT / MUL_MAT_ID on the CUDA backend for n = 1..nmax columns.
//
// Run twice with different GGML_CUDA_MMVQ_MAX (8 = MMVQ up to its cap, 0 = always MMQ) and compare;
// aragosta/tools/tune_mmq_crossover.py does exactly that and derives per-quant crossover tables.
//
//   mmq-sweep --type q4_K --m 4096 --k 4096 --nmax 16
//   mmq-sweep --type q4_K --m 512 --k 2048 --experts 256 --used 8 --nmax 16      (MUL_MAT_ID)
//
// Output: one JSON object per n on stdout: {"type":..,"op":..,"n":..,"us":..}

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct args {
    std::string type    = "q4_K";
    int64_t     m       = 4096;
    int64_t     k       = 4096;
    int         nmax    = 16;
    int         experts = 0;  // >0 -> MUL_MAT_ID
    int         used    = 8;
    int         reps    = 0;  // independent weight tensors per graph (defeats L2 reuse); 0 = auto
    int         iters   = 200;
    std::string backend = "CUDA0";
};

static ggml_type parse_type(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && s == n) {
            return (ggml_type) t;
        }
    }
    fprintf(stderr, "unknown type '%s'\n", s.c_str());
    exit(2);
}

// fill a quantized tensor with quantized random data (random raw bytes would produce NaN/Inf scales)
static void fill_quant(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t k = t->ne[0];
    const int64_t rows = ggml_nrows(t);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> src(k * std::min<int64_t>(rows, 64));
    for (auto & v : src) {
        v = nd(rng);
    }
    std::vector<float> imat;
    if (ggml_quantize_requires_imatrix(t->type)) {
        imat.assign(k, 1.0f);
    }
    ggml_quantize_init(t->type);
    const size_t row_size = ggml_row_size(t->type, k);
    std::vector<uint8_t> buf(row_size * rows);
    // quantize 64 distinct rows once, tile them over the tensor
    const int64_t nq = std::min<int64_t>(rows, 64);
    ggml_quantize_chunk(t->type, src.data(), buf.data(), 0, nq, k, imat.empty() ? nullptr : imat.data());
    for (int64_t r = nq; r < rows; ++r) {
        memcpy(buf.data() + r * row_size, buf.data() + (r % nq) * row_size, row_size);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

int main(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : (fprintf(stderr, "missing value\n"), exit(2), nullptr); };
        if      (!strcmp(argv[i], "--type"))    a.type    = next();
        else if (!strcmp(argv[i], "--m"))       a.m       = atoll(next());
        else if (!strcmp(argv[i], "--k"))       a.k       = atoll(next());
        else if (!strcmp(argv[i], "--nmax"))    a.nmax    = atoi(next());
        else if (!strcmp(argv[i], "--experts")) a.experts = atoi(next());
        else if (!strcmp(argv[i], "--used"))    a.used    = atoi(next());
        else if (!strcmp(argv[i], "--reps"))    a.reps    = atoi(next());
        else if (!strcmp(argv[i], "--iters"))   a.iters   = atoi(next());
        else if (!strcmp(argv[i], "--backend")) a.backend = next();
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    const ggml_type type = parse_type(a.type);
    const bool      moe  = a.experts > 0;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(a.backend.c_str());
    if (!dev) { fprintf(stderr, "backend %s not found\n", a.backend.c_str()); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    // keep the working set well above L2: ~96 MiB of weights per graph
    const size_t w_bytes = ggml_row_size(type, a.k) * a.m * (moe ? a.experts : 1);
    const int reps = a.reps > 0 ? a.reps : (int) std::clamp<size_t>((96u << 20) / std::max<size_t>(w_bytes, 1), 1, 16);

    std::mt19937 rng(1234);
    for (int n = 1; n <= a.nmax; ++n) {
        ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (reps * 3 + 8) + ggml_graph_overhead() + (1u << 20), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = moe ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a.k, 1, n)
                              : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a.k, n);
        ggml_tensor * ids = moe ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, a.used, n) : nullptr;
        std::vector<ggml_tensor *> ws, outs;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (int r = 0; r < reps; ++r) {
            ggml_tensor * w = moe ? ggml_new_tensor_3d(ctx, type, a.k, a.m, a.experts)
                                  : ggml_new_tensor_2d(ctx, type, a.k, a.m);
            ws.push_back(w);
            ggml_tensor * o = moe ? ggml_mul_mat_id(ctx, w, x, ids) : ggml_mul_mat(ctx, w, x);
            outs.push_back(o);
            ggml_build_forward_expand(gf, o);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) { fprintf(stderr, "alloc failed (n=%d)\n", n); return 1; }

        std::vector<float> xv(ggml_nelements(x));
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : xv) v = nd(rng);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        if (moe) {
            std::vector<int32_t> iv(a.used * n), perm(a.experts);
            for (int t = 0; t < n; ++t) {
                std::iota(perm.begin(), perm.end(), 0);
                std::shuffle(perm.begin(), perm.end(), rng);
                std::copy(perm.begin(), perm.begin() + a.used, iv.begin() + t * a.used);
            }
            ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
        }
        for (auto * w : ws) fill_quant(w, rng);

        for (int i = 0; i < 10; ++i) ggml_backend_graph_compute_async(be, gf);
        ggml_backend_synchronize(be);
        // best of 5 timed batches: robust against clock ramp / background noise
        double best = 1e30;
        for (int b = 0; b < 5; ++b) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < a.iters; ++i) ggml_backend_graph_compute_async(be, gf);
            ggml_backend_synchronize(be);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, us / ((double) a.iters * reps));
        }
        printf("{\"type\":\"%s\",\"op\":\"%s\",\"m\":%lld,\"k\":%lld,\"n\":%d,\"us\":%.3f}\n", a.type.c_str(),
               moe ? "mm_id" : "mm", (long long) a.m, (long long) a.k, n, best);
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
