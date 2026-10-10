// Teacher-forced per-step logits replay around a rejection, WITH the MoE expert cache armed.
//
// Based on archive/work/strix-halo/qsa-item4/mstep.cpp.  The original mstep never armed the
// expert-cache arena (that happens in common.cpp, not when a context is created directly), so it
// only ever exercised the streaming host-expert path.  This variant adds the `-ncmoe` host-expert
// overrides and calls llama_model_moe_cache_preflight() before the first context, exactly as the
// CLI/server do (common/common.cpp), so the cache-band routed-expert MMVQ take-over is live.
//
// Method (same as mstep):
//   pass 1 (ctx1): greedy decode width 1 for N steps; record T[s] and H1[s] = hash(logits that
//                  predicted T[s]) with context = prompt + T[0..s-1].
//   pass 2 (ctx2): replay T in W-token batches.  With RB>0 and JUNK=1 the batch is
//                  [T[k], junk, junk, ...] : row 0 is the accepted token, rows 1..W-1 are the
//                  rejected draft proposals.  After the batch, seq_rm() rolls the rejected tokens
//                  back (a rejection), so the next batch starts one token later.  Row j (non-junk)
//                  logits must equal H1[k+j+1] from the plain run.
//
//   A mismatch therefore means: after decoding the batch and rolling it back, the state the next
//   step reads is NOT the state the plain run had -- the recurrent/KV rewind restored the wrong
//   state.  No mismatch means the rollback replay is pure (for this config).
//
// env:
//   W=4            verify batch width (n_max+1)
//   N=150          number of greedy steps
//   RB=3           rollback per step (accepted=1); JUNK=1 substitutes junk for rows 1..W-1
//   JUNK=1         feed unrelated tokens for the rolled-back rows (the rejection)
//   RS=3           llama_context_params.n_rs_seq (MTP: n_max)
//   NCMOE=48       host-expert offload (0 = all experts on CPU, <n> = first n layers)
//   CACHE=1        call llama_model_moe_cache_preflight (default: 1 when NCMOE>0)
//   AUX_MIB=4096   auxiliary reserve passed to the preflight
//   SPLIT=tensor|layer|row, CTK, CTV, CTX=4096, NGL=99, POUT=1, TAIL=0
//   QUIET=1        silence the model loader
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <string>
#include <vector>

static int argmax(const float * a, int n) { int b = 0; for (int i = 1; i < n; ++i) if (a[i] > a[b]) b = i; return b; }
static uint64_t fh(const float * a, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * p = (const unsigned char *) a;
    for (size_t i = 0; i < (size_t) n * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static ggml_type kvt(const char * n) {
    if (!n) return GGML_TYPE_F16;
    if (!strcmp(n,"f16"))   return GGML_TYPE_F16;
    if (!strcmp(n,"q8_0"))  return GGML_TYPE_Q8_0;
    if (!strcmp(n,"bf16"))  return GGML_TYPE_BF16;
    if (!strcmp(n,"q4_0"))  return GGML_TYPE_Q4_0;
    if (!strcmp(n,"q4_1"))  return GGML_TYPE_Q4_1;
    if (!strcmp(n,"q5_0"))  return GGML_TYPE_Q5_0;
    return GGML_TYPE_F16;
}

static void quiet_log(ggml_log_level, const char *, void *) {}

static int g_nom = 0;
static int g_tail = 0;
static int g_pout = 1;

static llama_context * mkctx(llama_model * model, int CTX, int ub, uint32_t rs, ggml_type tk, ggml_type tv) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = CTX; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.n_rs_seq = rs;
    if (g_nom > 0) { cp.n_outputs_max = (uint32_t) g_nom; cp.n_outputs_max_per_seq = (uint32_t) g_nom; }
    cp.type_k = tk; cp.type_v = tv;
    return llama_init_from_model(model, cp);
}

static bool prefill(llama_context * ctx, const std::vector<llama_token> & toks, int P, int ub) {
    llama_batch b = llama_batch_init(ub, 0, 1);
    const int main_end = P - g_tail;
    for (int i = 0; i < main_end; ++i) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens] = (i >= P - g_pout) ? 1 : 0;
        b.n_tokens++;
        if (b.n_tokens == ub) {
            if (llama_decode(ctx, b) != 0) return false;
            b.n_tokens = 0;
        }
    }
    if (b.n_tokens > 0 && llama_decode(ctx, b) != 0) { llama_batch_free(b); return false; }
    b.n_tokens = 0;
    for (int i = main_end; i < P; ++i) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens] = (i >= P - g_pout) ? 1 : 0;
        b.n_tokens++;
    }
    if (b.n_tokens > 0 && llama_decode(ctx, b) != 0) { llama_batch_free(b); return false; }
    llama_batch_free(b);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model text P ubatch\n", argv[0]); return 1; }
    const char * mpath = argv[1]; const char * tpath = argv[2];
    const int P = atoi(argv[3]); const int ub = atoi(argv[4]);
    const int W = getenv("W") ? atoi(getenv("W")) : 4;
    const int N = getenv("N") ? atoi(getenv("N")) : 150;
    const uint32_t rs  = getenv("RS")  ? (uint32_t) atoi(getenv("RS"))  : 0;
    const uint32_t rs1 = getenv("RS1") ? (uint32_t) atoi(getenv("RS1")) : rs;
    const uint32_t rs2 = getenv("RS2") ? (uint32_t) atoi(getenv("RS2")) : rs;
    g_nom = getenv("NOM") ? atoi(getenv("NOM")) : 0;
    g_tail = getenv("TAIL") ? atoi(getenv("TAIL")) : 0;
    g_pout = getenv("POUT") ? atoi(getenv("POUT")) : 1;
    const int CTX = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    const ggml_type tk = kvt(getenv("CTK"));
    const ggml_type tv = getenv("CTV") ? kvt(getenv("CTV")) : tk;

    if (getenv("QUIET")) llama_log_set(quiet_log, nullptr);

    ggml_backend_load_all(); llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    if (getenv("NGL")) mp.n_gpu_layers = atoi(getenv("NGL"));

    // host-expert overrides (-ncmoe); same shape as tests/test-logits-width-probe.cpp
    static std::list<std::string> ov_strs;
    static std::vector<llama_model_tensor_buft_override> ov;
    const char * ncmoe_env = getenv("NCMOE");
    const int n_cpu_moe = ncmoe_env ? atoi(ncmoe_env) : -1;
    if (n_cpu_moe == 0) {
        ov.push_back({ "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type() });
        ov.push_back({ nullptr, nullptr });
        mp.tensor_buft_overrides = ov.data();
    } else if (n_cpu_moe > 0) {
        for (int i = 0; i < n_cpu_moe; ++i) {
            ov_strs.push_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
            ov.push_back({ ov_strs.back().c_str(), ggml_backend_cpu_buffer_type() });
        }
        ov.push_back({ nullptr, nullptr });
        mp.tensor_buft_overrides = ov.data();
    }

    if (const char * sp = getenv("SPLIT")) {
        if      (!strcmp(sp, "tensor")) mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        else if (!strcmp(sp, "row"))    mp.split_mode = LLAMA_SPLIT_MODE_ROW;
        else if (!strcmp(sp, "layer"))  mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    }

    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);

    // arm the expert cache exactly like common.cpp does: after the model is loaded, before the
    // first context exists.  Without this a standalone probe only exercises the streaming path.
    const bool cache = getenv("CACHE") ? atoi(getenv("CACHE")) != 0 : (n_cpu_moe != 0);
    if (cache) {
        const char * am = getenv("AUX_MIB");
        const size_t aux = (size_t) (am ? atoll(am) : 4096) * 1024 * 1024;
        llama_model_moe_cache_preflight(model, aux);
    }

    std::string text;
    { FILE * f = fopen(tpath, "rb"); if (!f) { perror("text"); return 1; }
      char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) text.append(b, r); fclose(f); }
    std::vector<llama_token> toks(262144);
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
    if (n < P) { fprintf(stderr, "text too short: %d < %d\n", n, P); return 1; }

    fprintf(stderr, "[rreplay] W=%d N=%d RB=%s JUNK=%s RS1=%u RS2=%u P=%d ub=%d CTX=%d split=%d ncmoe=%d cache=%d\n",
            W, N, getenv("RB") ? getenv("RB") : "0", getenv("JUNK") ? getenv("JUNK") : "0",
            rs1, rs2, P, ub, CTX, (int) mp.split_mode, n_cpu_moe, (int) cache);

    // ---- pass 1: greedy at width 1 ----
    std::vector<llama_token> T; std::vector<uint64_t> H1;
    {
        llama_context * c = mkctx(model, CTX, ub, rs1, tk, tv);
        if (!c) { fprintf(stderr, "ctx1 init failed\n"); return 1; }
        if (!prefill(c, toks, P, ub)) { fprintf(stderr, "ctx1 prefill failed\n"); return 1; }
        const float * l = llama_get_logits_ith(c, -1);
        T.push_back((llama_token) argmax(l, nv)); H1.push_back(fh(l, nv));
        llama_batch b = llama_batch_init(1, 0, 1);
        for (int s = 0; s + 1 < N; ++s) {
            b.n_tokens = 1;
            b.token[0] = T[s]; b.pos[0] = P + s;
            b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            if (llama_decode(c, b) != 0) { fprintf(stderr, "ctx1 decode failed at %d\n", s); return 1; }
            const float * ll = llama_get_logits_ith(c, 0);
            T.push_back((llama_token) argmax(ll, nv)); H1.push_back(fh(ll, nv));
        }
        llama_batch_free(b);
        llama_free(c);
    }
    { uint64_t th = 1469598103934665603ULL;
      for (size_t i = 0; i < T.size(); ++i) { const unsigned char * pp = (const unsigned char *) &T[i];
        for (size_t b2 = 0; b2 < sizeof(llama_token); ++b2) { th ^= pp[b2]; th *= 1099511628211ULL; } }
      fprintf(stderr, "[rreplay] generated %zu tokens (pass1) Thash=%016llx\n", T.size(), (unsigned long long) th); }

    // ---- pass 2: replay T in batches of W with rejection rollback ----
    int mismatches = 0;
    {
        llama_context * c = mkctx(model, CTX, ub, rs2, tk, tv);
        if (!c) { fprintf(stderr, "ctx2 init failed\n"); return 1; }
        if (!prefill(c, toks, P, ub)) { fprintf(stderr, "ctx2 prefill failed\n"); return 1; }
        {   // the prefill's last position predicts T[0]
            const float * l = llama_get_logits_ith(c, -1);
            if (fh(l, nv) != H1[0]) {
                printf("MISMATCH pos=%d (prefill) h1=%016llx hW=%016llx\n", P,
                       (unsigned long long) H1[0], (unsigned long long) fh(l, nv));
                mismatches++;
            }
        }
        llama_batch b = llama_batch_init(W, 0, 1);
        const int RB = getenv("RB") ? atoi(getenv("RB")) : 0;
        const int step = (RB > 0) ? W - RB : W;
        if (step < 1) { fprintf(stderr, "bad RB=%d (W=%d)\n", RB, W); return 1; }
        const bool junk = getenv("JUNK") != nullptr && atoi(getenv("JUNK")) != 0;
        for (int k = 0; k + W <= N - 1; k += step) {
            b.n_tokens = W;
            for (int j = 0; j < W; ++j) {
                b.token[j] = (junk && j > 0) ? (llama_token) ((T[k + j] + 1) % nv) : T[k + j];
                b.pos[j] = P + k + j;
                b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1;
            }
            if (llama_decode(c, b) != 0) { fprintf(stderr, "ctx2 decode failed at %d\n", k); return 1; }
            for (int j = 0; j < W; ++j) {
                if (junk && j > 0) continue;
                const int idx = k + j + 1;
                const float * l = llama_get_logits_ith(c, j);
                const uint64_t h = fh(l, nv);
                if (h != H1[idx]) {
                    if (mismatches < 40) printf("MISMATCH pos=%d (k=%d j=%d) h1=%016llx hW=%016llx\n",
                            P + idx, k, j, (unsigned long long) H1[idx], (unsigned long long) h);
                    mismatches++;
                }
            }
            if (RB > 0) {
                llama_memory_t mem = llama_get_memory(c);
                if (!llama_memory_seq_rm(mem, 0, P + k + step, -1)) {
                    fprintf(stderr, "seq_rm(%d..) false at k=%d\n", P + k + step, k);
                }
            }
        }
        llama_batch_free(b);
        llama_free(c);
    }

    if (cache) {
        int64_t hits = 0, misses = 0, arena = 0;
        if (llama_moe_cache_stats(model, &hits, &misses, &arena)) {
            fprintf(stderr, "[rreplay] cache hits=%lld misses=%lld arena=%.1f MiB\n",
                    (long long) hits, (long long) misses, (double) arena / (1024.0*1024.0));
        }
    }

    printf("[rreplay] W=%d RB=%s JUNK=%s P=%d mismatches=%d %s\n",
           W, getenv("RB") ? getenv("RB") : "0", getenv("JUNK") ? getenv("JUNK") : "0",
           P, mismatches, mismatches ? "IMPURE" : "PURE");
    llama_model_free(model); llama_backend_free();
    return mismatches ? 2 : 0;
}
