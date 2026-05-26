#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "block_gemm_cuda.h"

#define BLOCK_SIZE 16

__global__ void blockGemm(float* A, float* B, float* C, size_t n) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    float sum = 0.0f;

    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        int ax = t * BLOCK_SIZE + tx;
        int by = t * BLOCK_SIZE + ty;

        sA[ty][tx] = (y < n && ax < n) ? A[y * n + ax] : 0.0f;
        sB[ty][tx] = (x < n && by < n) ? B[by * n + x] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }
    
    if (y < n && x < n) {
        C[y * n + x] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);

    const float* aptr = a.data();
    const float* bptr = b.data();
    float* cptr = c.data();

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int bytes = n * n * sizeof(float);
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);
    
    cudaMemcpy(A, aptr, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, bptr, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    blockGemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n);

    cudaMemcpy(cptr, C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return c;
}

#if 0
std::vector<float> NaiveGemmRef(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.f);

    const float* aptr = a.data();
    const float* bptr = b.data();
    float* cptr = c.data();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float aval = aptr[i * n + k];
            for (int j = 0; j < n; j++) {
                cptr[i * n + j] += aval * bptr[k * n + j];
            }
        }
    }

    return c;
}

int main() {
    size_t n = 4096;
    std::vector<float> a(n*n);
    std::vector<float> b(n*n);
    for (size_t i = 0; i < n*n; i++) {
        a[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
        b[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto c = BlockGemmCUDA(a, b, n);

    auto cref = NaiveGemmRef(a, b, n);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(c[i] - cref[i]));
    }
    printf("max absolute error = %.5g\n", err);
    
    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = BlockGemmCUDA(a, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.2f\n", time);

    return 0;
}
#endif
