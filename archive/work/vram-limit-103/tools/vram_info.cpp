// Print device 0's free/total VRAM as the HIP runtime sees it (hipMemGetInfo).
//   vram_info            -> "free 32566 total 32624"
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    hipSetDevice(0);
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) { fprintf(stderr, "hipMemGetInfo failed\n"); return 1; }
    printf("free %zu total %zu\n", free_b / 1048576, total_b / 1048576);
    return 0;
}
