// Minimal deterministic probe for the multi-sequence rollback bug (delivery TODO #52).
//
// Scenario (mirrors llama-server -np 2 --kv-unified --spec-type draft-mtp with two
// concurrent greedy requests): sequence 1 runs a speculative verify batch of K
// tokens and then a rejection rollback of r tokens, while a *different* sequence
// (0) runs a > n_rs_batch prefill that shares the same llama_process call / ubatch.
//
// We compare the logits sequence 1 produces when replaying the rolled-back tokens
// under three schedules, all on the same context shape:
//
//   ref : seq 1 alone (prefill -> verify K -> rollback r -> replay)
//   ms  : seq 1 verify and seq 0 prefill placed in the SAME batch
//   sep : seq 1 verify and seq 0 prefill in SEPARATE llama_process calls
//
//   ms != ref  => cross-sequence batching corrupts seq 1's rollback
//   sep != ref => merely running another sequence's prefill changes seq 1
//
// Both contexts use n_seq_max = 2 so the recurrent cell layout is comparable.

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static llama_context * make_ctx(
        const common_params & params, llama_model * model,
        uint32_t n_rs_seq, uint32_t n_rs_batch, uint32_t n_ubatch,
        uint32_t n_seq_max, uint32_t n_ctx, bool unified) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.n_rs_batch = n_rs_batch;
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_ctx;
    cparams.n_ubatch   = n_ubatch;
    cparams.kv_unified = unified;
    return llama_init_from_model(model, cparams);
}

static bool decode_batch(llama_context * ctx, const std::vector<llama_token> & toks,
                         llama_pos pos0, uint32_t n, llama_seq_id seq, bool last_logits) {
    common_batch batch(ctx);
    for (uint32_t i = 0; i < n; ++i) {
        batch.add(toks[pos0 + i], pos0 + i, seq, last_logits && i + 1 == n);
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq) {
    common_batch batch(ctx);
    batch.add(tok, pos, seq, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static std::vector<llama_token> make_tokens(
        llama_context * ctx, const llama_vocab * vocab, uint32_t n_tokens, uint32_t n_vocab) {
    std::vector<llama_token> toks;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        for (uint32_t i = 0; i < n_tokens; ++i) {
            toks.push_back((llama_token) ((7*i + 3) % n_vocab));
        }
    } else {
        auto toks0 = common_tokenize(ctx, "The quick brown fox jumps over the lazy dog", true);
        if (toks0.empty()) {
            toks0 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        }
        for (uint32_t i = 0; i < n_tokens; ++i) {
            toks.push_back(toks0[i % toks0.size()] + (llama_token) ((i / toks0.size()) % 3));
        }
    }
    return toks;
}

// one arm; returns {verify last-token logits, replay first-token logits} for sequence 1
static void run_arm(
        const common_params & params, llama_model * model, llama_context * ctx_tok,
        const llama_vocab * vocab,
        uint32_t n_rs_seq, uint32_t n_rs_batch, uint32_t n_ubatch,
        uint32_t n_pre, uint32_t K, uint32_t r, uint32_t n_fill,
        int mode /* 0=ref 1=same-batch 2=separate */,
        std::vector<float> & verify_out, std::vector<float> & replay_out) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks = make_tokens(ctx_tok, vocab, n_pre + K + 4, (uint32_t) n_vocab);
    std::vector<llama_token> fill = make_tokens(ctx_tok, vocab, n_fill + 4, (uint32_t) n_vocab);
    // shift the filler so it is not identical to seq 1's stream
    for (auto & t : fill) { t = (llama_token) ((t + 7) % n_vocab); }
    if (getenv("MSQ_SAMEFILL")) { // control: seq 0 carries the same tokens as seq 1's verify batch
        for (uint32_t i = 0; i < n_fill; ++i) { fill[i] = toks[n_pre + (i % K)]; }
    }

    const bool uni = getenv("MSQ_UNI") ? atoi(getenv("MSQ_UNI")) != 0 : true;
    llama_context * ctx = make_ctx(params, model, n_rs_seq, n_rs_batch, n_ubatch, 2, 1024, /*unified=*/uni);
    if (ctx == nullptr) {
        fprintf(stderr, "  failed to init context\n");
        return;
    }

    if (n_pre > 0 && !decode_batch(ctx, toks, 0, n_pre, 1, true)) {
        fprintf(stderr, "  seq1 prefill failed\n");
        llama_free(ctx);
        return;
    }

    // seq 0 prefill (only for the modes that involve it)
    if (mode == 2 && !decode_batch(ctx, fill, 0, n_fill, 0, false)) {
        fprintf(stderr, "  seq0 prefill failed\n");
        llama_free(ctx);
        return;
    }

    // seq 1 verify batch (K tokens), optionally with seq 0's prefill in the same batch
    {
        common_batch batch(ctx);
        if (mode == 3) {
            for (uint32_t i = 0; i < n_fill; ++i) {
                batch.add(fill[i], i, 0, false);
            }
        }
        for (uint32_t i = 0; i < K; ++i) {
            batch.add(toks[n_pre + i], n_pre + i, 1, true);
        }
        if (mode == 1) {
            for (uint32_t i = 0; i < n_fill; ++i) {
                batch.add(fill[i], i, 0, false);
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "  seq1 verify failed\n");
            llama_free(ctx);
            return;
        }
        verify_out.clear();
        for (uint32_t i = 0; i < K; ++i) {
            // in mode 3 seq 0's tokens come first in the batch, so seq 1's outputs are offset
            const int32_t bi = (mode == 3 ? (int32_t) n_fill : 0) + (int32_t) i;
            const float * lv = llama_get_logits_ith(ctx, bi);
            if (lv == nullptr) {
                fprintf(stderr, "  missing verify logits\n");
                llama_free(ctx);
                return;
            }
            verify_out.insert(verify_out.end(), lv, lv + n_vocab);
        }
    }

    const uint32_t p0 = n_pre + K - r;
    if (n_rs_seq == 0) {
        // no snapshot path; only the raw forward is comparable
        llama_free(ctx);
        return;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, (llama_pos) p0, -1)) {
        fprintf(stderr, "  rollback refused (n_pre=%u K=%u r=%u)\n", n_pre, K, r);
        llama_free(ctx);
        return;
    }

    for (uint32_t i = p0; i < n_pre + K; ++i) {
        if (!decode_one(ctx, toks[i], (llama_pos) i, 1)) {
            fprintf(stderr, "  replay failed at %u\n", i);
            llama_free(ctx);
            return;
        }
        const float * l = llama_get_logits_ith(ctx, 0);
        if (l == nullptr) {
            fprintf(stderr, "  missing logits\n");
            llama_free(ctx);
            return;
        }
        if (i == p0) {
            replay_out.assign(l, l + n_vocab);
        }
    }

    llama_free(ctx);
}

static double diff_max(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return -1.0;
    }
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs((double) a[i] - (double) b[i]));
    }
    return d;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict     = 1;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "failed to init model\n");
        return 1;
    }
    if ((!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) && !getenv("MSQ_FORCE")) {
        fprintf(stderr, "skipping for non-recurrent model (MSQ_FORCE=1 to override)\n");
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    constexpr double eps = 1e-5;

    int n_fail = 0;

    // the production shape: n_rs_seq = 3 (draft n_max = 3), n_rs_batch = 4
    const uint32_t n_rs_seq   = getenv("MSQ_RS")   ? (uint32_t) atoi(getenv("MSQ_RS"))   : 3;
    const uint32_t n_rs_batch = getenv("MSQ_NB")   ? (uint32_t) atoi(getenv("MSQ_NB"))   : 4;
    const uint32_t n_ubatch   = getenv("MSQ_UB") ? (uint32_t) atoi(getenv("MSQ_UB")) : 128;
    const uint32_t n_pre      = getenv("MSQ_PRE") ? (uint32_t) atoi(getenv("MSQ_PRE")) : 64;

    if (getenv("MSQ_ARM")) {
        // single arm in isolation (separate process), print a digest of the verify logits
        const uint32_t K    = (uint32_t) atoi(getenv("MSQ_K"));
        const uint32_t r    = (uint32_t) atoi(getenv("MSQ_R"));
        const uint32_t fl   = (uint32_t) atoi(getenv("MSQ_FILL"));
        const uint32_t npre = getenv("MSQ_PRE") ? (uint32_t) atoi(getenv("MSQ_PRE")) : 64;
        const int      mode = atoi(getenv("MSQ_ARM"));
        std::vector<float> v, rp;
        run_arm(params, model, llama_init->context(), vocab,
                n_rs_seq, n_rs_batch, n_ubatch, npre, K, r, fl, mode, v, rp);
        uint64_t h = 1469598103934665603ull;
        for (float x : v) {
            uint32_t b;
            memcpy(&b, &x, sizeof(b));
            h = (h ^ b) * 1099511628211ull;
        }
        fprintf(stderr, "ARM %d K=%u r=%u fill=%u pre=%u verify_digest=%016llx n=%zu\n",
                mode, K, r, fl, npre, (unsigned long long) h, v.size());
        return 0;
    }

    for (uint32_t K : { 4u, 5u, 8u }) {
        for (uint32_t r = 1; r < K; ++r) {
            for (uint32_t n_fill : { 32u, 64u, 128u }) {
                if (getenv("MSQ_K")) {
                    if (K != (uint32_t) atoi(getenv("MSQ_K"))) continue;
                    if (r != (uint32_t) atoi(getenv("MSQ_R"))) continue;
                    if (n_fill != (uint32_t) atoi(getenv("MSQ_FILL"))) continue;
                }
                std::vector<float> vref, ref, vms, ms, vsep, sep;
                run_arm(params, model, llama_init->context(), vocab,
                        n_rs_seq, n_rs_batch, n_ubatch, n_pre, K, r, n_fill, 0, vref, ref);
                run_arm(params, model, llama_init->context(), vocab,
                        n_rs_seq, n_rs_batch, n_ubatch, n_pre, K, r, n_fill, 1, vms, ms);
                run_arm(params, model, llama_init->context(), vocab,
                        n_rs_seq, n_rs_batch, n_ubatch, n_pre, K, r, n_fill, 2, vsep, sep);
                const double d_ms   = diff_max(ref, ms);
                const double d_sep  = diff_max(ref, sep);
                const double dv_ms  = diff_max(vref, vms);
                const double dv_sep = diff_max(vref, vsep);
                // per verify token diff
                std::string per;
                if ((int) vref.size() == n_vocab * (int) K) {
                    for (uint32_t i = 0; i < K; ++i) {
                        double d = 0.0;
                        for (int t = 0; t < n_vocab; ++t) {
                            d = std::max(d, std::fabs((double) vref[i*n_vocab + t] - (double) vms[i*n_vocab + t]));
                        }
                        per += std::to_string((int) (d * 1000));
                        per += " ";
                    }
                }
                const char * verdict = (d_ms < 0 || d_sep < 0) ? "ERROR"
                                    : (d_ms > eps ? "MS-DIFF" : (d_sep > eps ? "SEP-DIFF" : "ok"));
                fprintf(stderr, "  K=%u r=%u n_fill=%3u  repl ms=%9.3g sep=%9.3g  verify ms=%9.3g sep=%9.3g  vt[%s] %s\n",
                        K, r, n_fill, d_ms, d_sep, dv_ms, dv_sep, per.c_str(), verdict);
                if (d_ms > eps || d_sep > eps || d_ms < 0 || d_sep < 0) {
                    n_fail++;
                }
            }
        }
    }

    fprintf(stderr, "\n%s: total failures = %d\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 2 : 0;
}
