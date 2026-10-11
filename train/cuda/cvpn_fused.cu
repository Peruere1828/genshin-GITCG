// MX550 (sm_75) fused kernels for the CVPN forward pass.
//
// The toy CVPN forward issues ~22 kernel launches per call; at the batch sizes
// the search/inference actually uses (B=1..64) it is launch- and memory-bound,
// not FLOP-bound (see scripts/bench_ops.py). These kernels fuse the elementwise
// + reduction parts so the same math runs in ~9 launches:
//
//   * layernorm_silu        token_proj epilogue (LayerNorm + SiLU): ~5 -> 1
//   * masked_mean_pool      masked token pooling:                  ~3 -> 1
//   * scorer_tail           option_scorer SiLU + last linear + mask: ~3 -> 1
//   * value_head            value MLP (Linear+SiLU+Linear):        ~3 -> 1
//
// GEMMs stay on cuBLAS (it wins for these sizes); only epilogues/reductions are
// rewritten. All kernels are fp32 (the MX550 has no tensor cores; fp16 is
// slower, see reports/explore/mx550_ops.md).
//
// Build via train/cuda_kernels.py (TORCH_CUDA_ARCH_LIST=7.5).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float silu(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

// Block-wide sum reduction over ``blockDim.x`` threads (any size <= 1024).
// A ``stride >>= 1`` tree is only correct for power-of-two blocks: with e.g.
// blockDim=96 the strides 48/24/12/6/3/1 leave shared[2]'s terms unfolded and
// silently drop them. This doubling form folds every partial for any block size
// (used by the fused kernels whose blockDim = min(1024, max(32, D))).
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = 1; stride < bdim; stride <<= 1) {
        int index = 2 * stride * tid;
        if (index < bdim && index + stride < bdim) {
            shared[index] += shared[index + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// y = silu(LayerNorm(x, w, b)) over the last dim D.  rows = product of leading dims.
__global__ void layernorm_silu_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ y, int rows, int D, float eps) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * D;
    float* yr = y + (size_t)row * D;
    float sum = 0.0f, sumsq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = xr[i];
        sum += v;
        sumsq += v * v;
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    sumsq = block_reduce_sum(sumsq, shared);
    float mean = sum / (float)D;
    float var = sumsq / (float)D - mean * mean;
    float rstd = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float n = (xr[i] - mean) * rstd * w[i] + b[i];
        yr[i] = silu(n);
    }
}

// out[b, d] = sum_t projected[b, t, d] * mask[b, t] / max(1, sum_t mask[b, t])
__global__ void masked_mean_pool_kernel(
    const float* __restrict__ projected, const unsigned char* __restrict__ mask,
    float* __restrict__ out, int B, int T, int D) {
    int b = blockIdx.x;
    if (b >= B) return;
    float denom = 0.0f;
    for (int t = 0; t < T; ++t) denom += mask[(size_t)b * T + t] ? 1.0f : 0.0f;
    if (denom < 1.0f) denom = 1.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T; ++t) {
            if (mask[(size_t)b * T + t]) acc += projected[((size_t)b * T + t) * D + d];
        }
        out[(size_t)b * D + d] = acc / denom;
    }
}

// logits[b, k] = (sum_h W2[h] * silu(hidden[b, k, h]) + b2) if mask else -inf
__global__ void scorer_tail_kernel(
    const float* __restrict__ hidden, const float* __restrict__ W2, float b2,
    const unsigned char* __restrict__ option_mask, float* __restrict__ logits,
    int B, int K, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * K) return;
    const float* hrow = hidden + (size_t)idx * H;
    float acc = b2;
    for (int h = 0; h < H; ++h) acc += W2[h] * silu(hrow[h]);
    logits[idx] = option_mask[idx] ? acc : -INFINITY;
}

// value[b] = sum_j W2[j] * silu(dot(W1[j], pooled[b]) + b1[j]) + b2
__global__ void value_head_kernel(
    const float* __restrict__ pooled, const float* __restrict__ W1,
    const float* __restrict__ b1, const float* __restrict__ W2, float b2,
    float* __restrict__ value, int B, int D, int VH) {
    extern __shared__ float shared[];
    int b = blockIdx.x;
    if (b >= B) return;
    const float* pr = pooled + (size_t)b * D;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < VH; j += blockDim.x) {
        const float* wj = W1 + (size_t)j * D;
        float dot = b1[j];
        for (int i = 0; i < D; ++i) dot += wj[i] * pr[i];
        acc += W2[j] * silu(dot);
    }
    acc = block_reduce_sum(acc, shared);
    if (threadIdx.x == 0) value[b] = acc + b2;
}

// Full token projection fusion: y = silu(LayerNorm(x @ W^T + bias)) per row.
// W (D x K) is staged in shared once per block and reused across BM rows, so it
// is read from global once per block instead of once per output.
constexpr int ROW_TILE = 16;

__global__ void linear_ln_silu_kernel(
    const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ bias,
    const float* __restrict__ lnw, const float* __restrict__ lnb, float* __restrict__ y,
    int rows, int K, int D, float eps) {
    extern __shared__ float shared[];
    float* sW = shared;            // D * K
    float* sRow = sW + (size_t)D * K;  // D
    float* sRed = sRow + D;        // blockDim scratch
    for (int i = threadIdx.x; i < D * K; i += blockDim.x) sW[i] = W[i];
    __syncthreads();
    for (int r = 0; r < ROW_TILE; ++r) {
        int row = blockIdx.x * ROW_TILE + r;
        bool active = row < rows;
        if (active) {
            const float* xr = x + (size_t)row * K;
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                const float* w = sW + (size_t)d * K;
                float acc = bias[d];
                for (int k = 0; k < K; ++k) acc += w[k] * xr[k];
                sRow[d] = acc;
            }
        }
        __syncthreads();
        if (active) {
            float sum = 0.0f, sumsq = 0.0f;
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                float v = sRow[i];
                sum += v;
                sumsq += v * v;
            }
            sum = block_reduce_sum(sum, sRed);
            __syncthreads();
            sumsq = block_reduce_sum(sumsq, sRed);
            float mean = sum / (float)D;
            float var = sumsq / (float)D - mean * mean;
            float rstd = rsqrtf(var + eps);
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                float n = (sRow[d] - mean) * rstd * lnw[d] + lnb[d];
                y[(size_t)row * D + d] = silu(n);
            }
        }
        __syncthreads();
    }
}

// y = silu(x @ W^T + bias) per row (option projection), W staged in shared.
__global__ void linear_silu_kernel(
    const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ bias,
    float* __restrict__ y, int rows, int K, int D) {
    extern __shared__ float sW[];
    for (int i = threadIdx.x; i < D * K; i += blockDim.x) sW[i] = W[i];
    __syncthreads();
    for (int r = 0; r < ROW_TILE; ++r) {
        int row = blockIdx.x * ROW_TILE + r;
        if (row >= rows) break;
        const float* xr = x + (size_t)row * K;
        float* yr = y + (size_t)row * D;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float* w = sW + (size_t)d * K;
            float acc = bias[d];
            for (int k = 0; k < K; ++k) acc += w[k] * xr[k];
            yr[d] = silu(acc);
        }
    }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace

torch::Tensor layernorm_silu(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be cuda float32");
    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto bc = b.contiguous();
    int D = xc.size(-1);
    int rows = (int)(xc.numel() / D);
    auto y = torch::empty_like(xc);
    int threads = std::min(1024, std::max(32, D));
    int smem = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    layernorm_silu_kernel<<<rows, threads, smem, stream>>>(
        xc.data_ptr<float>(), wc.data_ptr<float>(), bc.data_ptr<float>(),
        y.data_ptr<float>(), rows, D, (float)eps);
    return y;
}

torch::Tensor masked_mean_pool(torch::Tensor projected, torch::Tensor token_mask) {
    TORCH_CHECK(projected.is_cuda() && projected.dtype() == torch::kFloat32, "projected must be cuda float32");
    TORCH_CHECK(token_mask.dtype() == torch::kBool, "token_mask must be bool");
    auto p = projected.contiguous();
    int B = p.size(0), T = p.size(1), D = p.size(2);
    auto mask = token_mask.contiguous();
    auto out = torch::empty({B, D}, p.options());
    int threads = std::min(256, std::max(32, D));
    auto stream = at::cuda::getCurrentCUDAStream();
    masked_mean_pool_kernel<<<B, threads, 0, stream>>>(
        p.data_ptr<float>(), reinterpret_cast<const unsigned char*>(mask.data_ptr<bool>()),
        out.data_ptr<float>(), B, T, D);
    return out;
}

torch::Tensor scorer_tail(torch::Tensor hidden, torch::Tensor W2, double b2, torch::Tensor option_mask) {
    TORCH_CHECK(hidden.is_cuda() && hidden.dtype() == torch::kFloat32, "hidden must be cuda float32");
    TORCH_CHECK(option_mask.dtype() == torch::kBool, "option_mask must be bool");
    auto h = hidden.contiguous();
    int B = h.size(0), K = h.size(1), H = h.size(2);
    auto w2 = W2.contiguous();
    auto mask = option_mask.contiguous();
    auto logits = torch::empty({B, K}, h.options());
    int total = B * K;
    int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();
    scorer_tail_kernel<<<ceil_div(total, threads), threads, 0, stream>>>(
        h.data_ptr<float>(), w2.data_ptr<float>(), (float)b2,
        reinterpret_cast<const unsigned char*>(mask.data_ptr<bool>()),
        logits.data_ptr<float>(), B, K, H);
    return logits;
}

torch::Tensor value_head(
    torch::Tensor pooled, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, double b2) {
    TORCH_CHECK(pooled.is_cuda() && pooled.dtype() == torch::kFloat32, "pooled must be cuda float32");
    auto p = pooled.contiguous();
    auto w1 = W1.contiguous();
    auto bb1 = b1.contiguous();
    auto w2 = W2.contiguous();
    int B = p.size(0), D = p.size(1), VH = w1.size(0);
    auto value = torch::empty({B}, p.options());
    int threads = std::min(1024, std::max(32, VH));
    int smem = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    value_head_kernel<<<B, threads, smem, stream>>>(
        p.data_ptr<float>(), w1.data_ptr<float>(), bb1.data_ptr<float>(),
        w2.data_ptr<float>(), (float)b2, value.data_ptr<float>(), B, D, VH);
    return value;
}

torch::Tensor token_proj_ln_silu(
    torch::Tensor x, torch::Tensor W, torch::Tensor bias, torch::Tensor lnw, torch::Tensor lnb,
    double eps) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be cuda float32");
    auto xc = x.contiguous();
    auto wc = W.contiguous();
    auto bc = bias.contiguous();
    auto lw = lnw.contiguous();
    auto lb = lnb.contiguous();
    int rows = xc.size(0), K = xc.size(1), D = wc.size(0);
    auto y = torch::empty({rows, D}, xc.options());
    int threads = std::min(1024, std::max(32, D));
    int smem = (D * K + D + threads) * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(linear_ln_silu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    linear_ln_silu_kernel<<<ceil_div(rows, ROW_TILE), threads, smem, stream>>>(
        xc.data_ptr<float>(), wc.data_ptr<float>(), bc.data_ptr<float>(),
        lw.data_ptr<float>(), lb.data_ptr<float>(), y.data_ptr<float>(), rows, K, D, (float)eps);
    return y;
}

torch::Tensor linear_silu(torch::Tensor x, torch::Tensor W, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be cuda float32");
    auto xc = x.contiguous();
    auto wc = W.contiguous();
    auto bc = bias.contiguous();
    int rows = xc.size(0), K = xc.size(1), D = wc.size(0);
    auto y = torch::empty({rows, D}, xc.options());
    int threads = std::min(256, std::max(32, D));
    int smem = D * K * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(linear_silu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    linear_silu_kernel<<<ceil_div(rows, ROW_TILE), threads, smem, stream>>>(
        xc.data_ptr<float>(), wc.data_ptr<float>(), bc.data_ptr<float>(), y.data_ptr<float>(),
        rows, K, D);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_silu", &layernorm_silu, "fused LayerNorm+SiLU");
    m.def("masked_mean_pool", &masked_mean_pool, "masked mean pool");
    m.def("scorer_tail", &scorer_tail, "fused scorer SiLU+linear+mask");
    m.def("value_head", &value_head, "fused value MLP");
    m.def("token_proj_ln_silu", &token_proj_ln_silu, "fused token projection+LayerNorm+SiLU");
    m.def("linear_silu", &linear_silu, "fused linear+SiLU");
}
