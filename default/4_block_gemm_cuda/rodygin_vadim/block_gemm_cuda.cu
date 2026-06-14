#include <cuda_runtime.h>
#include <vector>
#include "block_gemm_cuda.h"

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices in output matrix
    int row = blockIdx.y * BLOCK_SIZE + ty;  // i
    int col = blockIdx.x * BLOCK_SIZE + tx;  // j

    float sum = 0.0f;
    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        int aCol = tile * BLOCK_SIZE + tx;  // Global column index
        if (row < n && aCol < n) {
            sA[ty][tx] = A[row * n + aCol];  // A[i][k]
        } else {
            sA[ty][tx] = 0.0f;
        }
        
        int bRow = tile * BLOCK_SIZE + ty;  // Global row index
        if (bRow < n && col < n) {
            sB[ty][tx] = B[bRow * n + col];  // B[k][j]
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int size = n * n;
    const size_t bytes = size * sizeof(float);

    std::vector<float> c(size);

    float* d_A;
    float* d_B;
    float* d_C;
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    blockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return c;
}

