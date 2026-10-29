// randomized check of llama_kv_cells::get_n_multi_seq() against a recount after every operation (Claude, 2026-10-29)
#include "llama-kv-cells.h"
#include <cstdio>
#include <random>
#include <algorithm>
static uint32_t recount(const llama_kv_cells & c) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < c.size(); ++i) n += c.seq_get_all(i).count() > 1;
    return n;
}
int main() {
    std::mt19937 rng(12345);
    const uint32_t N = 512; const int S = 6;
    llama_kv_cells c; c.resize(N);
    long ops = 0, multi_seen = 0;
    for (int it = 0; it < 400000; ++it) {
        const uint32_t i = rng() % N; const llama_seq_id s = rng() % S;
        switch (rng() % 11) {
            case 0: if (c.is_empty(i)) { c.pos_set(i, rng() % 1000); c.seq_add(i, s); } break;
            case 1: if (!c.is_empty(i) && !c.seq_has(i, s)) c.seq_add(i, s); break;
            case 2: if (!c.is_empty(i) && c.seq_has(i, s)) c.seq_rm(i, s); break;
            case 3: if (!c.is_empty(i)) c.seq_keep(i, s); break;
            case 4: if (!c.is_empty(i)) c.rm(i); break;
            case 5: { const uint32_t j = rng() % N; if (!c.is_empty(i) && c.is_empty(j)) c.mv(i, j); } break;
            case 6: if (!c.is_empty(i)) { c.pos_add(i, (int) (rng() % 200) - 150); c.reset_shift(); } break;  // the cache applies and resets shifts before any state copy
            case 7: { const uint32_t a = rng() % (N - 32); auto cp = c.cp(a, 32); if (recount(cp) != cp.get_n_multi_seq()) { printf("cp counter wrong\n"); return 1; }
                      const uint32_t b = rng() % (N - 32); c.set(b, cp); } break;
            case 8: if (rng() % 2000 == 0) c.reset(); break;
            case 9: { std::vector<uint32_t> a, b; for (int k = 0; k < 16; ++k) { a.push_back(rng() % N); }
                      std::sort(a.begin(), a.end()); a.erase(std::unique(a.begin(), a.end()), a.end());
                      while (b.size() < a.size()) { const uint32_t x = rng() % N; if (std::find(b.begin(), b.end(), x) == b.end()) b.push_back(x); }
                      auto cp = c.cp(a); if (recount(cp) != cp.get_n_multi_seq()) { printf("cp(idxs) counter wrong\n"); return 1; }
                      c.set(b, cp); } break;
            case 10: if (!c.is_empty(i)) { c.pos_div(i, 2); c.reset_shift(); } break;
        }
        ++ops;
        const uint32_t r = recount(c);
        multi_seen += r > 0;
        if (r != c.get_n_multi_seq()) { printf("MISMATCH after %ld ops: counter %u recount %u\n", ops, c.get_n_multi_seq(), r); return 1; }
    }
    printf("ok: %ld random operations, counter == recount every time (%ld states with shared cells)\n", ops, multi_seen);
    return 0;
}
