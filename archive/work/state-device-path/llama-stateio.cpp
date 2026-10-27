// one-off measurement harness for TODO #110: time the LLAMA_STATE_SEQ_FLAGS_ON_DEVICE save/restore path
// on a fragmented (interleaved-slot) KV cache.  Not part of the delivery.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf n_tokens [n_threads]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int    n_tokens   = atoi(argv[2]);
    const int    n_threads  = argc > 3 ? atoi(argv[3]) : 8;
    const bool   frag       = argc > 4 ? strcmp(argv[4], "frag") == 0 : true;
    const llama_seq_id sid  = 1;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.split_mode   = LLAMA_SPLIT_MODE_TENSOR;
    mparams.main_gpu     = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 2*n_tokens + 64;
    cparams.n_batch     = 256;
    cparams.n_ubatch    = 256;
    cparams.n_seq_max   = 2;
    cparams.kv_unified  = true;
    cparams.n_threads   = n_threads;
    cparams.no_perf     = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // interleave the two sequences one token at a time so seq 1's cells are maximally fragmented
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed*1664525u + 1013904223u; return (llama_token) (seed % (uint32_t) n_vocab); };

    std::vector<llama_token> toks0(n_tokens), toks1(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        toks0[i] = rnd();
        toks1[i] = rnd();
    }

    const double t_dec0 = now_ms();
    for (int i = 0; i < n_tokens; ++i) {
        llama_batch b = llama_batch_init(frag ? 2 : 1, 0, 2);
        if (frag) {
            b.n_tokens = 2;
            b.token[0] = toks0[i]; b.pos[0] = i; b.seq_id[0][0] = 0; b.n_seq_id[0] = 1; b.logits[0] = 1;
            b.token[1] = toks1[i]; b.pos[1] = i; b.seq_id[1][0] = 1; b.n_seq_id[1] = 1; b.logits[1] = 1;
        } else {
            b.n_tokens = 1;
            b.token[0] = toks1[i]; b.pos[0] = i; b.seq_id[0][0] = 1; b.n_seq_id[0] = 1; b.logits[0] = 1;
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "decode failed at %d\n", i);
            return 1;
        }
        llama_batch_free(b);
    }
    const double t_dec1 = now_ms();
    printf("decode: %d tokens (%s) in %.0f ms\n", n_tokens, frag ? "interleaved 2 seqs" : "contiguous 1 seq", t_dec1 - t_dec0);

    const uint32_t flags = getenv("HOSTIO") ? LLAMA_STATE_SEQ_FLAGS_NONE : LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

    const size_t size = llama_state_seq_get_size_ext(ctx, sid, flags);
    printf("%s state size: %.1f MiB\n", getenv("HOSTIO") ? "host" : "device", size / 1048576.0);

    std::vector<uint8_t> data(size);

    // save
    double best_save = 1e9, best_load = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now_ms();
        const size_t n = llama_state_seq_get_data_ext(ctx, data.data(), size, sid, flags);
        const double t1 = now_ms();
        if (n != size) {
            fprintf(stderr, "save size mismatch %zu != %zu\n", n, size);
            return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        const double t2 = now_ms();
        const size_t m = llama_state_seq_set_data_ext(ctx, data.data(), size, sid, flags);
        const double t3 = now_ms();
        if (m != size) {
            fprintf(stderr, "load size mismatch %zu != %zu\n", m, size);
            return 1;
        }
        printf("rep %d: save %.2f ms (%.2f GiB/s), restore %.2f ms (%.2f GiB/s)\n",
               rep, t1 - t0, size / 1073741824.0 / ((t1 - t0)/1e3),
               t3 - t2, size / 1073741824.0 / ((t3 - t2)/1e3));
        if (t1 - t0 < best_save) best_save = t1 - t0;
        if (t3 - t2 < best_load) best_load = t3 - t2;
    }
    printf("BEST: save %.2f ms, restore %.2f ms, total %.2f ms (%.2f GiB/s)\n",
           best_save, best_load, best_save + best_load,
           size / 1073741824.0 / ((best_save + best_load)/1e3));

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
