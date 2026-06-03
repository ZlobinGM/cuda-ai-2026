#include "gelu_cuda.h"
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <cuda/cmath>

__global__ void geluKernel(float* res, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = input[idx];
    float targ = 0.797885*(x+0.044715*x*x*x);
    float th = 1 - 2. / (cuda::std::expf(2*targ)+1);
    if(idx < n)
        res[idx] = 0.5*x*(1.+th);
}

struct MemoryKeeper
{
    float* input_ptr = nullptr;
    float* res_ptr = nullptr;
    ~MemoryKeeper();
    int64_t memory_allocated = 0;
} memoryKeeper;

MemoryKeeper* getMemoryKeeper(int size)
{
    if(memoryKeeper.memory_allocated < size)
    {
        if(memoryKeeper.memory_allocated > 0 )
        {
            cudaFree(memoryKeeper.input_ptr);
            cudaFree(memoryKeeper.res_ptr);
        }
        cudaMalloc(&memoryKeeper.input_ptr, size);
        cudaMalloc(&memoryKeeper.res_ptr, size);
        memoryKeeper.memory_allocated = size;
    }
    return &memoryKeeper;
}

MemoryKeeper::~MemoryKeeper()
{
    if(memory_allocated != 0)
    {
        cudaFree(input_ptr);
        cudaFree(res_ptr);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    const int n = input.size();
    const size_t size = n * sizeof(float);

    float* input_ptr = getMemoryKeeper(size)->input_ptr;
    float* res_ptr = getMemoryKeeper(size)->res_ptr;

    std::vector<float> res(n);

    cudaMemcpy(input_ptr, input.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(res_ptr, input_ptr, n);

    cudaMemcpy(res.data(), res_ptr, size, cudaMemcpyDeviceToHost);
    
    return res;
}