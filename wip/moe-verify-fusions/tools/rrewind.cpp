// Single-context recurrent-rewind probe.
//
// At every position we decode the SAME token twice from the SAME starting state:
//   path P (plain):    seq_rm back to pos k, decode [t_k]            -> L_plain, state S_plain
//   path V (verify):   seq_rm back to pos k, decode [t_k, j,j,j]     -> L_verify(row 0)
//                      then seq_rm back to pos k+1 (the MTP rejection rollback)
//   then compare L_verify(row 0) == L_plain, and hash(S_verify) == hash(S_plain).
//
// A mismatch isolates the divergence to the *forward* (plain vs W-token verify arithmetic) and/or the
// rollback restore, independent of a separate plain reference context (which is what made the
// pass1/pass2 replay ambiguous when the cache state drifts between the two contexts).
//
// env: W, N, RS=3, NCMOE, CACHE, MOE_EXPERT_CACHE_MIB, SPLIT, CTK, CTV, CTX, P.  See rollback-replay.cpp.
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
static uint64_t bh(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
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

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model text P ubatch\n", argv[0]); return 1; }
    const char * mpath = argv[1]; const char * tpath = argv[2];
    const int P = atoi(argv[3]); const int ub = atoi(argv[4]);
    const int W = getenv("W") ? atoi(getenv("W")) : 4;
    const int N = getenv("N") ? atoi(getenv("N")) : 60;
    const uint32_t rs = getenv("RS") ? (uint32_t) atoi(getenv("RS")) : 3;
    const int CTX = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    const ggml_type tk = kvt(getenv("CTK"));
    const ggml_type tv = getenv("CTV") ? kvt(getenv("CTV")) : tk;
    if (W < 2) { fprintf(stderr, "W must be >= 2\n"); return 1; }

    if (getenv("QUIET")) llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all(); llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 99;

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
    if (n < P) { fprintf(stderr, "text too short\n"); return 1; }

    fprintf(stderr, "[rrewind] W=%d N=%d RS=%u P=%d ub=%d split=%d ncmoe=%d cache=%d mib=%s\n",
            W, N, rs, P, ub, (int) mp.split_mode, n_cpu_moe, (int) cache,
            getenv("MOE_EXPERT_CACHE_MIB") ? getenv("MOE_EXPERT_CACHE_MIB") : "auto");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = CTX; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.n_rs_seq = rs;
    cp.type_k = tk; cp.type_v = tv;
    llama_context * c = llama_init_from_model(model, cp);
    if (!c) { fprintf(stderr, "ctx init failed\n"); return 1; }

    // prefill
    {
        llama_batch b = llama_batch_init(ub, 0, 1);
        for (int i = 0; i < P; ++i) {
            b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i;
            b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = (i == P - 1) ? 1 : 0;
            b.n_tokens++;
            if (b.n_tokens == ub) { if (llama_decode(c, b)) { fprintf(stderr, "prefill fail\n"); return 1; } b.n_tokens = 0; }
        }
        if (b.n_tokens && llama_decode(c, b)) { fprintf(stderr, "prefill fail\n"); return 1; }
        llama_batch_free(b);
    }

    std::vector<uint8_t> st1, st2;
    const bool do_state = getenv("STATE") != nullptr && atoi(getenv("STATE")) != 0;
    const size_t ssz = do_state ? llama_state_seq_get_size(c, 0) : 0;
    if (do_state) { st1.resize(ssz); st2.resize(ssz); }

    llama_batch b1 = llama_batch_init(1, 0, 1);
    llama_batch bW = llama_batch_init(W, 0, 1);
    const int POLL = getenv("POLLUTE") ? atoi(getenv("POLLUTE")) : 0;
    llama_batch bP = llama_batch_init(POLL > 0 ? POLL : 1, 0, 1);
    llama_memory_t mem = llama_get_memory(c);

    // first token = argmax of the prefill logits
    const float * lp = llama_get_logits_ith(c, -1);
    llama_token ck = (llama_token) argmax(lp, nv);

    int bad_logit = 0, bad_state = 0;
    for (int k = 0; k < N; ++k) {
        const llama_pos p = P + k;
        // path P: plain decode of ck from the state at p
        b1.n_tokens = 1;
        b1.token[0] = ck; b1.pos[0] = p; b1.n_seq_id[0] = 1; b1.seq_id[0][0] = 0; b1.logits[0] = 1;
        if (llama_decode(c, b1)) { fprintf(stderr, "plain decode fail k=%d\n", k); return 1; }
        const uint64_t hL = fh(llama_get_logits_ith(c, 0), nv);
        const llama_token nxt = (llama_token) argmax(llama_get_logits_ith(c, 0), nv);
        if (do_state) llama_state_seq_get_data(c, st1.data(), st1.size(), 0);
        // back to the state before ck
        if (!llama_memory_seq_rm(mem, 0, p, -1)) { fprintf(stderr, "seq_rm P fail k=%d\n", k); return 1; }

        // optional: emulate the MTP draft's cache activity between the two paths.  The tokens are
        // rolled back, so the recurrent state is unchanged -- only the expert-cache resident set moves.
        if (POLL > 0) {
            bP.n_tokens = POLL;
            for (int j = 0; j < POLL; ++j) {
                bP.token[j] = (llama_token) ((ck * 131 + j * 1000003 + 7) % nv);
                bP.pos[j] = p + j; bP.n_seq_id[j] = 1; bP.seq_id[j][0] = 0; bP.logits[j] = 0;
            }
            if (llama_decode(c, bP)) { fprintf(stderr, "pollute decode fail k=%d\n", k); return 1; }
            if (!llama_memory_seq_rm(mem, 0, p, -1)) { fprintf(stderr, "seq_rm POLL fail k=%d\n", k); return 1; }
        }

        // path V: verify batch [ck, junk...] from the same state
        bW.n_tokens = W;
        for (int j = 0; j < W; ++j) {
            bW.token[j] = j == 0 ? ck : (llama_token) ((ck + 1 + j) % nv);
            bW.pos[j] = p + j; bW.n_seq_id[j] = 1; bW.seq_id[j][0] = 0; bW.logits[j] = 1;
        }
        if (llama_decode(c, bW)) { fprintf(stderr, "verify decode fail k=%d\n", k); return 1; }
        const uint64_t hV = fh(llama_get_logits_ith(c, 0), nv);
        // the MTP rejection rollback: keep only ck
        if (!llama_memory_seq_rm(mem, 0, p + 1, -1)) { fprintf(stderr, "seq_rm V fail k=%d\n", k); return 1; }
        if (do_state) llama_state_seq_get_data(c, st2.data(), st2.size(), 0);

        if (hL != hV) {
            if (bad_logit < 20) printf("LOGIT MISMATCH k=%d pos=%d plain=%016llx verify=%016llx\n",
                    k, p, (unsigned long long) hL, (unsigned long long) hV);
            bad_logit++;
        }
        if (do_state && bh(st1.data(), st1.size()) != bh(st2.data(), st2.size())) {
            if (bad_state < 20) printf("STATE MISMATCH k=%d pos=%d\n", k, p);
            bad_state++;
        }
        ck = nxt;
    }

    if (cache) {
        int64_t hits = 0, misses = 0, arena = 0;
        if (llama_moe_cache_stats(model, &hits, &misses, &arena))
            fprintf(stderr, "[rrewind] cache hits=%lld misses=%lld arena=%.1f MiB\n",
                    (long long) hits, (long long) misses, (double) arena / (1024.0*1024.0));
    }
    printf("[rrewind] W=%d N=%d logit_mismatches=%d state_mismatches=%d %s\n",
           W, N, bad_logit, bad_state, (bad_logit || bad_state) ? "IMPURE" : "PURE");

    llama_batch_free(b1); llama_batch_free(bW); llama_batch_free(bP); llama_free(c);
    llama_model_free(model); llama_backend_free();
    return (bad_logit || bad_state) ? 2 : 0;
}
