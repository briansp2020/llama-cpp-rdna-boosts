// Hold a fixed amount of VRAM on every visible HIP device, so a 32 GiB card presents the
// free-memory figure of a smaller one.
//   vram_fill <MiB per device> [seconds] [chunk MiB]
//
// chunk MiB (default 0 = one BO) splits the hold into that many MiB per hipMalloc.  The
// granularity matters: the kernel's TTM layer evicts a *whole* BO, so a single 16 GiB BO that
// must give up 500 MiB is migrated to system memory in full, while a chunked hold gives up only
// what is needed.  That is what separates "the driver moved 16 GiB during the run" from "the run
// is intrinsically slow under memory pressure" (TODO #103).
//
// VRAM_FILL_TOUCH=1 keeps the hold *live*: it rewrites the first MiB of every BO in a loop, the
// way a desktop compositor touches its buffers -- enough to keep the pages hot, far from enough
// to saturate the memory bus.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <utility>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

int main(int argc, char ** argv) {
    const size_t mib       = argc > 1 ? (size_t) atoll(argv[1]) : 8192;
    const int    secs      = argc > 2 ? atoi(argv[2]) : 0; // 0 = until killed
    const size_t chunk_mib = argc > 3 ? (size_t) atoll(argv[3]) : 0;
    const size_t chunk     = (chunk_mib ? chunk_mib : mib) << 20;
    const bool   touch     = getenv("VRAM_FILL_TOUCH") != nullptr;
    int n = 0;
    CK(hipGetDeviceCount(&n));
    printf("vram_fill: %d devices, %zu MiB each, chunk %zu MiB, touch %d\n",
           n, mib, chunk >> 20, (int) touch);

    std::vector<std::pair<char *, size_t>> keep;
    for (int i = 0; i < n; i++) {
        CK(hipSetDevice(i));
        size_t held = 0;
        while (held < mib << 20) {
            const size_t sz = std::min<size_t>(chunk, (mib << 20) - held);
            void * p = nullptr;
            hipError_t e = hipMalloc(&p, sz);
            if (e != hipSuccess) { fprintf(stderr, "hipMalloc(%zu MiB) failed: %s\n", sz >> 20, hipGetErrorString(e)); break; }
            CK(hipMemset(p, 0, sz));   // touch it so the pages are really resident
            keep.push_back({(char *) p, sz});
            held += sz;
        }
        size_t free_b = 0, total_b = 0;
        CK(hipMemGetInfo(&free_b, &total_b));
        printf("vram_fill: dev %d held %zu MiB in %zu BOs, free now %zu MiB of %zu MiB\n",
               i, held >> 20, keep.size(), free_b >> 20, total_b >> 20);
    }
    fflush(stdout);
    if (secs > 0) { sleep(secs); return 0; }
    if (touch) {
        for (;;) {
            for (auto & b : keep) {
                CK(hipMemset(b.first, 0, std::min<size_t>(1 << 20, b.second)));
            }
            usleep(2000);
        }
    }
    for (;;) { pause(); }
}
