// Chapter 19: complete FP32 GEMM experiment; CUDA compilation/run remain pending.
// NVIDIA CUDA Runtime is required. No cuBLAS dependency, no fabricated timings.
// Build on the target GPU host, e.g. nvcc -O3 -std=c++17 -arch=sm_80 gemm_bench.cu -o gemm_bench
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                             \
            throw std::runtime_error(std::string(#call) + ": " +                \
                                     cudaGetErrorString(error));                \
        }                                                                       \
    } while (false)

constexpr int TILE = 32;

__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc = fmaf(A[static_cast<size_t>(row) * K + k],
                   B[static_cast<size_t>(k) * N + col], acc);
    C[static_cast<size_t>(row) * N + col] = acc;
}

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
        int a_col = kt * TILE + tx;
        int b_row = kt * TILE + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[static_cast<size_t>(row) * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[static_cast<size_t>(b_row) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
}

// Each of the first 256 threads loads four elements. Unaligned/tail rows
// fall back to masked scalar loads; no barrier is hidden inside this helper.
__device__ __forceinline__ void load_tile4(const float* src, float* dst,
                                          int rows, int cols,
                                          int base_row, int base_col, int tid) {
    if (tid >= TILE * TILE / 4) return;
    int r = tid / (TILE / 4);
    int c = (tid % (TILE / 4)) * 4;
    int row = base_row + r, col = base_col + c;
    int local = r * TILE + c;
    if (row < rows && col + 3 < cols && cols % 4 == 0) {
        const float4 value = *reinterpret_cast<const float4*>(
            src + static_cast<size_t>(row) * cols + col);
        dst[local + 0] = value.x;
        dst[local + 1] = value.y;
        dst[local + 2] = value.z;
        dst[local + 3] = value.w;
    } else {
        for (int j = 0; j < 4; ++j)
            dst[local + j] = (row < rows && col + j < cols)
                ? src[static_cast<size_t>(row) * cols + col + j] : 0.0f;
    }
}

__global__ void gemm_vectorized(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    __shared__ float As[TILE * TILE];
    __shared__ float Bs[TILE * TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * TILE + tx;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
        load_tile4(A, As, M, K, blockIdx.y * TILE, kt * TILE, tid);
        load_tile4(B, Bs, K, N, kt * TILE, blockIdx.x * TILE, tid);
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            acc = fmaf(As[ty * TILE + k], Bs[k * TILE + tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
}

// Same 32x32 tile and vectorized copies, but 16x16 threads, four outputs each.
// No promise that occupancy or speed improves: measure registers/stalls/time.
__global__ __launch_bounds__(256) void gemm_tiled_2x2(
    const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE * TILE];
    __shared__ float Bs[TILE * TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;
    int r0 = blockIdx.y * TILE + 2 * ty;
    int c0 = blockIdx.x * TILE + 2 * tx;
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
        load_tile4(A, As, M, K, blockIdx.y * TILE, kt * TILE, tid);
        load_tile4(B, Bs, K, N, kt * TILE, blockIdx.x * TILE, tid);
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    acc[i][j] = fmaf(As[(2 * ty + i) * TILE + k],
                                      Bs[k * TILE + 2 * tx + j], acc[i][j]);
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            if (r0 + i < M && c0 + j < N)
                C[static_cast<size_t>(r0 + i) * N + c0 + j] = acc[i][j];
}

struct DeviceBuffer {
    float* data = nullptr;
    explicit DeviceBuffer(size_t elements) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), elements * sizeof(float)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void launch(int version, const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(version == 0 || version == 3 ? 16 : 32,
               version == 0 || version == 3 ? 16 : 32);
    int tile = version == 0 ? 16 : TILE;
    dim3 grid((N + tile - 1) / tile, (M + tile - 1) / tile);
    switch (version) {
        case 0: gemm_naive<<<grid, block>>>(A, B, C, M, N, K); break;
        case 1: gemm_tiled<<<grid, block>>>(A, B, C, M, N, K); break;
        case 2: gemm_vectorized<<<grid, block>>>(A, B, C, M, N, K); break;
        case 3: gemm_tiled_2x2<<<grid, block>>>(A, B, C, M, N, K); break;
        default: throw std::runtime_error("unknown GEMM version");
    }
    CUDA_CHECK(cudaGetLastError());
}

struct Validation {
    size_t checked;
    double max_abs;
    bool full;
};

Validation validate(const std::vector<float>& A, const std::vector<float>& B,
                    const std::vector<float>& C, int M, int N, int K,
                    double atol, double rtol) {
    for (float value : C)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite or unwritten output");
    bool full = static_cast<double>(M) * N * K <= 64000000.0;
    std::vector<size_t> indices;
    if (full) {
        indices.resize(C.size());
        for (size_t i = 0; i < C.size(); ++i) indices[i] = i;
    } else {
        indices = {0, static_cast<size_t>(N - 1), static_cast<size_t>(M - 1) * N, C.size() - 1};
        std::mt19937 rng(7);
        for (int i = 0; i < 128; ++i) indices.push_back(rng() % C.size());
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }
    double max_abs = 0.0;
    for (size_t index : indices) {
        int row = static_cast<int>(index / N), col = static_cast<int>(index % N);
        double ref = 0.0;
        for (int k = 0; k < K; ++k)
            ref += static_cast<double>(A[static_cast<size_t>(row) * K + k]) *
                   B[static_cast<size_t>(k) * N + col];
        double error = std::abs(static_cast<double>(C[index]) - ref);
        max_abs = std::max(max_abs, error);
        if (error > atol + rtol * std::abs(ref))
            throw std::runtime_error("CPU reference mismatch at output index " + std::to_string(index));
    }
    return {indices.size(), max_abs, full};
}

float benchmark(int version, const DeviceBuffer& A, const DeviceBuffer& B,
                DeviceBuffer& C, int M, int N, int K, int warmup, int repeat) {
    for (int i = 0; i < warmup; ++i) launch(version, A.data, B.data, C.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) launch(version, A.data, B.data, C.data, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (!(total_ms > 0.0f)) throw std::runtime_error("invalid event elapsed time");
    return total_ms / repeat;
}

int main(int argc, char** argv) {
    try {
        int M = 128, N = 128, K = 128, warmup = 5, repeat = 20;
        double atol = 1e-3, rtol = 1e-4;
        bool check_only = false;
        std::string selected = "all";
        const std::vector<std::string> names = {"naive", "tiled", "vectorized", "coarsened"};
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "--m M --n N --k K --warmup 5 --repeat 20 --atol 1e-3 --rtol 1e-4 "
                             "--version all|naive|tiled|vectorized|coarsened --check-only\n";
                return 0;
            }
            if (key == "--check-only") { check_only = true; continue; }
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            std::string value = argv[++i];
            if (key == "--m") M = std::stoi(value);
            else if (key == "--n") N = std::stoi(value);
            else if (key == "--k") K = std::stoi(value);
            else if (key == "--warmup") warmup = std::stoi(value);
            else if (key == "--repeat") repeat = std::stoi(value);
            else if (key == "--atol") atol = std::stod(value);
            else if (key == "--rtol") rtol = std::stod(value);
            else if (key == "--version") selected = value;
            else throw std::runtime_error("unknown argument " + key);
        }
        if (M <= 0 || N <= 0 || K <= 0 || M > 8192 || N > 8192 || K > 8192 ||
            warmup < 0 || repeat <= 0 || !std::isfinite(atol) || !std::isfinite(rtol) ||
            atol < 0 || rtol < 0)
            throw std::runtime_error("invalid dimensions/repetition/tolerance; dimensions must be 1..8192");
        if (selected != "all" && std::find(names.begin(), names.end(), selected) == names.end())
            throw std::runtime_error("unknown --version " + selected);

        int device = 0, driver = 0, runtime = 0;
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::cout << "device=" << properties.name << " sm=" << properties.major << properties.minor
                  << " driver_api=" << driver << " runtime=" << runtime << " seed=0 "
                  << "shape=" << M << 'x' << N << 'x' << K << " dtype=fp32 "
                  << "warmup=" << warmup << " repeat=" << repeat << " atol=" << atol << " rtol=" << rtol
                  << " cache=warm statistic=event_mean\n";

        std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N);
        std::vector<float> C(static_cast<size_t>(M) * N);
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        for (float& value : A) value = uniform(rng);
        for (float& value : B) value = uniform(rng);
        DeviceBuffer d_A(A.size()), d_B(B.size()), d_C(C.size());
        CUDA_CHECK(cudaMemcpy(d_A.data, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.data, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int version = 0; version < 4; ++version) {
            if (selected != "all" && selected != names[version]) continue;
            // NaN poison catches missing output stores before any timing is accepted.
            CUDA_CHECK(cudaMemset(d_C.data, 0xff, C.size() * sizeof(float)));
            launch(version, d_A.data, d_B.data, d_C.data, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(C.data(), d_C.data, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
            Validation result = validate(A, B, C, M, N, K, atol, rtol);
            std::cout << names[version] << " validation=" << (result.full ? "full" : "sampled")
                      << " checked=" << result.checked << " max_abs=" << result.max_abs;
            if (!check_only) {
                float ms = benchmark(version, d_A, d_B, d_C, M, N, K, warmup, repeat);
                double gflops = 2.0 * M * N * K / (ms * 1e6);
                std::cout << " ms=" << std::setprecision(8) << ms << " gflops=" << gflops;
            }
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
