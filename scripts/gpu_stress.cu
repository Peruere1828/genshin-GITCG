// GPU stress test (self-contained CUDA, no cuBLAS/PyTorch) — WSL/Linux side.
//
// Keeps the GPU busy so clocks/power can be watched, and prints sustained FP32
// GFLOPS. On the WSL MX550 the GPU runs full speed when the laptop is on AC and
// the battery is full/not charging (see reports/LOCAL_BASELINE.md).
//
// Build:  nvcc -O3 -arch=sm_75 -o gpu_stress scripts/gpu_stress.cu
// Run:    ./gpu_stress [seconds]     (default 60; Ctrl-C to stop early)
//
// Watch in parallel:
//   LD_LIBRARY_PATH=/usr/lib/wsl/lib nvidia-smi \
//     --query-gpu=pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit --format=csv -l 1
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void burn(float *sink, unsigned long long iters) {
    float a = (float)threadIdx.x * 0.001f + 1.0f;
    float b = (float)blockIdx.x * 0.001f + 2.0f;
    float c = 3.14159f;
    for (unsigned long long i = 0; i < iters; ++i) {
        a = fmaf(a, b, c);
        b = fmaf(b, c, a);
        c = fmaf(c, a, b);
    }
    if (a == 123.456f) sink[0] = a + b + c;  // never true: keeps the loop alive
}

static inline void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    int seconds = (argc > 1) ? std::atoi(argv[1]) : 60;

    int devs = 0;
    check(cudaGetDeviceCount(&devs), "cudaGetDeviceCount");
    if (devs == 0) { std::fprintf(stderr, "no CUDA device\n"); return 1; }
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("Device: %s (cc %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    float *sink;
    check(cudaMalloc(&sink, sizeof(float)), "cudaMalloc");
    const int blocks = prop.multiProcessorCount * 64;
    const int threads = 256;
    const unsigned long long iters = 400000;  // per launch

    burn<<<blocks, threads>>>(sink, 2000);  // warmup
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, now;
    cudaEventCreate(&start);
    cudaEventCreate(&now);
    cudaEventRecord(start);

    std::printf("Running %d blocks x %d threads for %d s ...\n", blocks, threads, seconds);
    double total_fma = 0.0;
    float elapsed_ms = 0.0f;
    int reps = 0;
    while (true) {
        burn<<<blocks, threads>>>(sink, iters);
        check(cudaDeviceSynchronize(), "sync");
        reps++;
        total_fma += (double)blocks * threads * iters * 3.0;
        cudaEventRecord(now);
        cudaEventSynchronize(now);
        cudaEventElapsedTime(&elapsed_ms, start, now);
        if (elapsed_ms >= seconds * 1000.0f) break;
        if (reps % 10 == 0)
            std::printf("  t=%5.1fs  %.1f GFLOP/s (fp32)\n",
                        elapsed_ms / 1000.0f, total_fma * 2.0 / elapsed_ms / 1e6);
    }
    std::printf("done: %d reps, %.1f s, avg %.1f GFLOP/s (fp32)\n",
                reps, elapsed_ms / 1000.0f, total_fma * 2.0 / elapsed_ms / 1e6);
    cudaFree(sink);
    return 0;
}
