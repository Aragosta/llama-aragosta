#!/usr/bin/env python3
"""One-shot source patch: aragosta scheduler hooks (prefill cap, FCFS, spec gating).

Kept as a script so the same edit can be re-applied on a fresh upstream checkout;
asserts every anchor matches exactly once, so it fails loudly when upstream moves.
"""
import sys


def rep(path, old, new):
    s = open(path).read()
    n = s.count(old)
    if n != 1:
        sys.exit(f"{path}: anchor matched {n}x: {old[:70]!r}")
    open(path, "w").write(s.replace(old, new))


rep("common/common.h",
    "    int32_t n_cache_reuse       = 0;     // min chunk size to reuse from the cache via KV shifting\n",
    "    int32_t n_cache_reuse       = 0;     // min chunk size to reuse from the cache via KV shifting\n"
    "    int32_t sched_prefill_chunk = 0;     // aragosta: max prompt tokens per step while any slot decodes; 0 = off\n"
    "    int32_t spec_max_active     = 0;     // aragosta: start no new drafts while more than N slots generate; 0 = off\n")

rep("common/arg.cpp",
    '    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_REUSE"));\n',
    '''    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_REUSE"));
    add_opt(common_arg(
        {"--sched-prefill-chunk"}, "N",
        string_format(
            "max prompt tokens scheduled per step while at least one slot is generating; bounds the decode\\n"
            "stall caused by long prefills (cf. vLLM long_prefill_token_threshold). 0 = fill n_batch (default: %d)",
            params.sched_prefill_chunk
        ),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("--sched-prefill-chunk must be >= 0");
            }
            params.sched_prefill_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SCHED_PREFILL_CHUNK"));
    add_opt(common_arg(
        {"--spec-max-active"}, "N",
        string_format(
            "start new speculative drafts only while at most N slots are generating; above that the verify\\n"
            "batch costs more than drafting saves. 0 = always draft (default: %d)",
            params.spec_max_active
        ),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("--spec-max-active must be >= 0");
            }
            params.spec_max_active = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_MAX_ACTIVE"));
''')

P = "tools/server/server-context.cpp"

rep(P, """        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;
""", """        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // aragosta: no new drafts at high concurrency (cf. vLLM disabling speculation by batch size):
        // the verify batch grows with users x (1 + n_draft) while the per-user gain shrinks.
        const bool spec_allow_new = params_base.spec_max_active <= 0 ||
            std::count_if(slots.begin(), slots.end(), [](const server_slot & s) {
                return s.state == SLOT_STATE_GENERATING;
            }) <= params_base.spec_max_active;
""")

rep(P, """                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(""", """                    } else if (spec_allow_new) {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(""")

rep(P, """        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {""", """        // aragosta: while any slot is generating, cap this step's prompt tokens so a long prefill
        // cannot stretch the step (and so every decoder's inter-token latency) to a full n_batch.
        // With no decoder the cap is lifted and an idle server prefills at full width.
        const int32_t n_batch_prompt = params_base.sched_prefill_chunk > 0 && !generating.empty()
            ? std::min(n_batch, (int32_t) batch.size() + params_base.sched_prefill_chunk)
            : n_batch;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            std::vector<server_slot *> order = prefill_order();
            iterate(order, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch_prompt) {""")

rep(P, "                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {",
       "                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch_prompt) {")

rep(P, "    void abort_all_slots(const std::string & reason) {", """    // aragosta: prompt order = task arrival (FCFS, as vLLM), parents before children.
    // Upstream fills in slot-index order, so a long prompt in a low slot delays later arrivals.
    std::vector<server_slot *> prefill_order() {
        std::vector<server_slot *> order;
        order.reserve(slots.size());
        for (auto & slot : slots) {
            order.push_back(&slot);
        }
        auto key = [](const server_slot * s) {
            const bool busy  = s->is_processing() && s->task;
            const bool child = busy && s->task->is_child();
            return std::make_tuple(!busy, child, busy ? s->task->id : 0, s->id);
        };
        std::stable_sort(order.begin(), order.end(), [&](const server_slot * a, const server_slot * b) {
            return key(a) < key(b);
        });
        return order;
    }

    void abort_all_slots(const std::string & reason) {""")
print("patched")
