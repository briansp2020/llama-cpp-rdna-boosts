// vram_probe2 <MiB_A> <MiB_B> — does the driver evict an already-touched buffer out of VRAM
// when the SAME process then allocates more, and if so, how fast is that buffer afterwards?
//
//   1. allocate A, touch it, measure the read bandwidth
//   2. allocate B, touch it, measure A's read bandwidth again
//
// A bandwidth collapse in step 2 is the over-commit mechanism behind TODO #103: the GPU is still
// "using" the buffer, it just reads it over PCIe from system memory.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

__global__ void read_sum(const float * __restrict__ p, size_t n, float * out) {
    float s = 0.f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) s += p[i];
    if (s == 12345.678f) out[0] = s;
}

static double read_bw(float * p, size_t n, float * out, hipEvent_t t0, hipEvent_t t1) {
    CK(hipEventRecord(t0));
    read_sum<<<320, 256>>>(p, n, out);
    CK(hipEventRecord(t1));
    CK(hipEventSynchronize(t1));
    float ms = 0; CK(hipEventElapsedTime(&ms, t0, t1));
    return (n * sizeof(float)) / (ms * 1e6);
}

int main(int argc, char ** argv) {
    const size_t a_mib = argc > 1 ? (size_t) atoll(argv[1]) : 8192;
    const size_t b_mib = argc > 2 ? (size_t) atoll(argv[2]) : 8192;
    hipSetDevice(0);
    size_t free_b = 0, total_b = 0;
    CK(hipMemGetInfo(&free_b, &total_b));
    printf("start: free %zu MiB of %zu MiB\n", free_b / 1048576, total_b / 1048576);

    float * A = nullptr; CK(hipMalloc(&A, a_mib << 20));
    CK(hipMemset(A, 0, a_mib << 20));
    float * d_out = nullptr; CK(hipMalloc(&d_out, 4));
    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    CK(hipMemGetInfo(&free_b, &total_b));
    printf("A=%zu MiB allocated, free %zu MiB; A read %.1f GB/s\n",
           a_mib, free_b / 1048576, read_bw(A, (a_mib << 20) / 4, d_out, t0, t1));

    float * B = nullptr;
    hipError_t e = hipMalloc(&B, b_mib << 20);
    if (e != hipSuccess) { printf("B hipMalloc(%zu MiB) FAILED: %s\n", b_mib, hipGetErrorString(e)); return 3; }
    CK(hipMemset(B, 0, b_mib << 20));
    CK(hipMemGetInfo(&free_b, &total_b));
    printf("B=%zu MiB allocated, free %zu MiB; A read %.1f GB/s, B read %.1f GB/s\n",
           b_mib, free_b / 1048576,
           read_bw(A, (a_mib << 20) / 4, d_out, t0, t1),
           read_bw(B, (b_mib << 20) / 4, d_out, t0, t1));
    for (int i = 0; i < 2; i++) {
        printf("A read %.1f GB/s, B read %.1f GB/s\n",
               read_bw(A, (a_mib << 20) / 4, d_out, t0, t1),
               read_bw(B, (b_mib << 20) / 4, d_out, t0, t1));
    }
    return 0;
}
