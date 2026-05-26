#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

#include "naive_gemm_cuda.h"

__global__ void NaiveGemmCUDAKernelBase(const float* a, const float* b, float* c, const int n, const int n_pow) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx >> n_pow;
    int j = idx & (n - 1);
    float res = 0.f;
    for (int k = 0; k < n; ++k) {
        res += a[i * n + k] * b[k * n + j];
    }
    c[idx] = res;
}

__global__ void NaiveGemmCUDAKernelBaseCheck(const float* a, const float* b, float* c, const int n, const int n_pow) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int i = idx >> n_pow;
        int j = idx & (n - 1);
        float res = 0.f;
        for (int k = 0; k < n; ++k) {
            res += a[i * n + k] * b[k * n + j];
        }
        c[idx] = res;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                     const std::vector<float>& b,
                                     int n) {
    const int size = n * n;
    const int memSize = size*sizeof(float);
    std::vector<float> c(size);
    uint block_size;
    if (size > 1024) {
        block_size = 1024;
    } else {
        block_size = 256;
    }
    const uint num_blocks = (size + block_size - 1) / block_size;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);
    int pow = static_cast<int>(std::log2(n));
    cudaMemcpy(d_a, a.data(), memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), memSize, cudaMemcpyHostToDevice);
    if (size % block_size == 0) {
        NaiveGemmCUDAKernelBase<<<num_blocks, block_size>>>(d_a, d_b, d_c, n, pow);
    } else {
        NaiveGemmCUDAKernelBaseCheck<<<num_blocks, block_size>>>(d_a, d_b, d_c, n, pow);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, memSize, cudaMemcpyDeviceToHost);
    return c;
}
