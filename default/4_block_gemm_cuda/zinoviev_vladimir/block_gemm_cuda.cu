#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

#include "block_gemm_cuda.h"

#define BLOCK_SIZE 16

__global__ void BlockGemmCUDAKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B[BLOCK_SIZE * BLOCK_SIZE];
    int idx_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx_col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int idx_shared = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int n_blocks = n / BLOCK_SIZE;
    int x_shift = blockIdx.x * BLOCK_SIZE;
    int y_step = BLOCK_SIZE * n;
    int y_shift = blockIdx.y * y_step;
    int idx_map = threadIdx.y * n + threadIdx.x;

    float res = 0.f;
    for (int block = 0; block < n_blocks; ++block) {
        int block_start_A = y_shift + block * BLOCK_SIZE;
        int block_start_B = block * y_step + x_shift;
        A[idx_shared] = a[block_start_A + idx_map];
        B[idx_shared] = b[block_start_B + idx_map];
        __syncthreads();
        for(int k = 0; k < BLOCK_SIZE; ++k) {
            res += A[threadIdx.y * BLOCK_SIZE + k] * B[k * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }
    c[idx_col * n + idx_row] = res;
}

__global__ void BlockGemmCUDAKernelCheck(const float* a, const float* b, float* c, int n) {
    __shared__ float A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B[BLOCK_SIZE * BLOCK_SIZE];
    int idx_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx_col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int idx_shared = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int x_shift = blockIdx.x * BLOCK_SIZE;
    int y_step = BLOCK_SIZE * n;
    int y_shift = blockIdx.y * y_step;
    int idx_map = threadIdx.y * n + threadIdx.x;

    float res = 0.f;
    for (int block = 0; block < n_blocks; ++block) {
        if (((block * BLOCK_SIZE + threadIdx.x) < n) && (idx_col < n)) {
            int block_start_A = y_shift + block * BLOCK_SIZE;
            A[idx_shared] = a[block_start_A + idx_map];
        } else {
            A[idx_shared] = 0.f;
        }
        if (((block * BLOCK_SIZE + threadIdx.y) < n) && (idx_row < n)) {
            int block_start_B = block * y_step + x_shift;
            B[idx_shared] = b[block_start_B + idx_map];
        } else {
            B[idx_shared] = 0.f;
        }

        __syncthreads();
        if (idx_row < n && idx_col < n) {
            for(int k = 0; k < BLOCK_SIZE; ++k) {
                res += A[threadIdx.y * BLOCK_SIZE + k] * B[k * BLOCK_SIZE + threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (idx_row < n && idx_col < n) {
        c[idx_col * n + idx_row] = res;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int size = n * n;
    const int memSize = size*sizeof(float);
    std::vector<float> c(size);
    constexpr uint block_size = BLOCK_SIZE;
    const uint num_blocks = (n + block_size - 1) / block_size;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);
    cudaMemcpy(d_a, a.data(), memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), memSize, cudaMemcpyHostToDevice);
    if (n % block_size == 0) {
        BlockGemmCUDAKernel<<<{num_blocks, num_blocks}, {block_size, block_size}, 2 * block_size * block_size>>>(d_a, d_b, d_c, n);
    } else {
        BlockGemmCUDAKernelCheck<<<{num_blocks, num_blocks}, {block_size, block_size}, 2 * block_size * block_size>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, memSize, cudaMemcpyDeviceToHost);

    return c;
}

