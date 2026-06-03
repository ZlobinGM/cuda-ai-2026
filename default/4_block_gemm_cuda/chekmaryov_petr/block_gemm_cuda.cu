#include "block_gemm_cuda.h"
#include <assert.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void gemmKernel(float* a_ptr, float* b_ptr, float* res_ptr, int n)
{
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int global_row = threadIdx.y + blockIdx.y * blockDim.y;
    int global_col = threadIdx.x + blockIdx.x * blockDim.x;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    __shared__ float block_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float4 block_b_[BLOCK_SIZE][BLOCK_SIZE/4];
    const float4* b4_ptr = reinterpret_cast<float4*>(b_ptr);
    for (int block = 0; block < gridDim.x; ++block) {
        block_a[local_row][(4 * local_col + 0)] = a_ptr[global_row * n + (block * BLOCK_SIZE + (4 * local_col + 0))];
        block_a[local_row][(4 * local_col + 1)] = a_ptr[global_row * n + (block * BLOCK_SIZE + (4 * local_col + 1))];
        block_a[local_row][(4 * local_col + 2)] = a_ptr[global_row * n + (block * BLOCK_SIZE + (4 * local_col + 2))];
        block_a[local_row][(4 * local_col + 3)] = a_ptr[global_row * n + (block * BLOCK_SIZE + (4 * local_col + 3))];
        block_b_[local_row][local_col] = b4_ptr[(block * BLOCK_SIZE + local_row) * n/4 + global_col];
        
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            float a = block_a[local_row][k]; 
            sum.x += a * block_b_[k][local_col].x;
            sum.y += a * block_b_[k][local_col].y;
            sum.z += a * block_b_[k][local_col].z;
            sum.w += a * block_b_[k][local_col].w;
        }
    __syncthreads();
    }
    if (global_col * 4 < n && global_row < n)
    {
        float4 accum = *(reinterpret_cast<float4*>(res_ptr) + global_row*n/4 + global_col);
        accum.x += sum.x;
        accum.y += sum.y;
        accum.z += sum.z;
        accum.w += sum.w;
        *(reinterpret_cast<float4*>(res_ptr) + global_row*n/4 + global_col) = accum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    assert(a.size() == n * n);
    assert(b.size() == n * n);
    const size_t size = n * n * sizeof(float);
    std::vector<float> res(n*n,0);

    float* a_ptr;
    float* b_ptr;
    float* res_ptr;
    cudaMalloc(&a_ptr, size);
    cudaMalloc(&b_ptr, size);
    cudaMalloc(&res_ptr, size);

    cudaMemcpy(a_ptr, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock((BLOCK_SIZE/4), BLOCK_SIZE);
    dim3 blockCount((n/4+(BLOCK_SIZE/4)-1) / (BLOCK_SIZE/4), (n+BLOCK_SIZE-1) / BLOCK_SIZE);
    gemmKernel<<<blockCount, threadsPerBlock>>>(a_ptr, b_ptr, res_ptr, n);
    cudaDeviceSynchronize();
    cudaMemcpy(res.data(), res_ptr, size, cudaMemcpyDeviceToHost);
    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(res_ptr);
    
    return res;
}