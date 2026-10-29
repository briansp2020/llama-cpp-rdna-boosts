// A new prompt cache buffer of n MiB as server_prompt_cache::take() makes it: mode 0 = resize(n) (the release), mode 1 = the
// patch's reserve(n), madvise(MADV_HUGEPAGE) on the 2 MiB-aligned part, resize(n).  Prints the time and AnonHugePages.
// g++ -O2 -o hugepage-bench hugepage-bench.cpp && ./hugepage-bench 3494 0 && ./hugepage-bench 3494 1   (Linux)
// Written by Claude (AI), 2026-10-29.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <sys/mman.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static long anon_huge_kb() { FILE * f = fopen("/proc/self/smaps_rollup", "r"); char l[256]; long v = -1; while (f && fgets(l, sizeof l, f)) if (sscanf(l, "AnonHugePages: %ld", &v) == 1) break; if (f) fclose(f); return v; }
int main(int argc, char ** argv) {
    const size_t n = (size_t) atof(argc > 1 ? argv[1] : "3494") << 20;
    const int mode = argc > 2 ? atoi(argv[2]) : 1;
    for (int rep = 0; rep < 3; ++rep) {
        double t0 = now();
        std::vector<uint8_t> buf;
        int rc = 0;
        if (mode) {
            buf.reserve(n);
            const uintptr_t huge = 2u << 20;
            const uintptr_t beg = ((uintptr_t) buf.data() + huge - 1) & ~(huge - 1);
            const uintptr_t end = ((uintptr_t) buf.data() + n) & ~(huge - 1);
            rc = madvise((void *) beg, end - beg, MADV_HUGEPAGE);
        }
        buf.resize(n);
        double t1 = now();
        printf("mode %d rc %d: %.0f MiB alloc+zero %.0f ms, AnonHugePages %ld MiB\n", mode, rc, n / 1048576.0, (t1 - t0) * 1e3, anon_huge_kb() / 1024);
    }
}
