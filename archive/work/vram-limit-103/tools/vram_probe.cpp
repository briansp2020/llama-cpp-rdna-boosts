// vram_probe <MiB> — allocate <MiB>, then measure write (memset) and read (reduce) bandwidth.
//
// Purpose (TODO #103): find out what the amdgpu/HIP allocator does when VRAM is nearly
// exhausted -- does hipMalloc fail, or does it succeed by placing the buffer outside VRAM
// (GTT / system memory), and if it succeeds, how fast can the GPU read it?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

__global__ void read_sum(const float * __restrict__ p, size_t n, float * out) {
    float s = 0.f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        s += p[i];
    }
    if (s == 12345.678f) out[0] = s; // never true; keeps the loop alive
}

__global__ void write_val(float * __restrict__ p, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        p[i] = 1.0f;
    }
}

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) atoll(argv[1]) : 2048;
    const size_t bytes = mib * 1024ull * 1024ull;
    const size_t n = bytes / sizeof(float);
    hipSetDevice(0);

    size_t free_b = 0, total_b = 0;
    CK(hipMemGetInfo(&free_b, &total_b));
    printf("before: free %zu MiB total %zu MiB\n", free_b / 1048576, total_b / 1048576);

    float * p = nullptr;
    hipError_t e = hipMalloc(&p, bytes);
    if (e != hipSuccess) { printf("hipMalloc(%zu MiB) FAILED: %s\n", mib, hipGetErrorString(e)); return 3; }
    CK(hipMemGetInfo(&free_b, &total_b));
    printf("hipMalloc(%zu MiB) ok, free now %zu MiB\n", mib, free_b / 1048576);

    float * d_out = nullptr;
    CK(hipMalloc(&d_out, 4));
    hipEvent_t t0, t1;
    CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));

    CK(hipMemset(p, 0, bytes));                    // first touch: real residency decision
    CK(hipDeviceSynchronize());
    CK(hipMemGetInfo(&free_b, &total_b));
    printf("after memset: free %zu MiB\n", free_b / 1048576);

    for (int it = 0; it < 3; it++) {
        CK(hipEventRecord(t0));
        CK(hipMemsetAsync(p, 0, bytes));
        CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
        float ms = 0; CK(hipEventElapsedTime(&ms, t0, t1));
        printf("write %zu MiB: %8.2f ms  %7.1f GB/s\n", mib, ms, bytes / (ms * 1e6));

        CK(hipEventRecord(t0));
        read_sum<<<80 * 4, 256>>>(p, n, d_out);
        CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
        CK(hipEventElapsedTime(&ms, t0, t1));
        printf("read  %zu MiB: %8.2f ms  %7.1f GB/s\n", mib, ms, bytes / (ms * 1e6));
    }
    return 0;
}
