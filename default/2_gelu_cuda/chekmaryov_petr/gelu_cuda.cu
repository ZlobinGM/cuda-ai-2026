#include "gelu_cuda.h"
#include <math.h>
#include <stdio.h>
#include <immintrin.h>

// CUDA-ядро для параллельного сложения элементов
__global__ void geluKernel(float* res, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = input[idx];
    float targ = 0.636619772368*(x+0.044715*x*x*x);
    float th = 1 - 2. / (std::exp(2*targ)+1);
    res[idx] = 0.5*x*(1.+th);
}

struct MemoryKeeper
{
    float* input_ptr = nullptr;
    float* res_ptr = nullptr;
    ~MemoryKeeper();
    int64_t memory_allocated = 0;
} memoryKeeper;

MemoryKeeper* getMemoryKeeper(int n)
{
    if(memoryKeeper.memory_allocated < n)
    {
        if(memoryKeeper.memory_allocated > 0 )
        {
            cudaFree(memoryKeeper.input_ptr);
            cudaFree(memoryKeeper.res_ptr);
        }
        cudaMalloc(&memoryKeeper.input_ptr, n);
        cudaMalloc(&memoryKeeper.res_ptr, n);
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
    int n = input.size();
    size_t size = n * sizeof(float);
    std::vector<float> res(n);

    float* input_ptr = nullptr;
    float* res_ptr = nullptr;

    cudaMemcpy(getMemoryKeeper(n)->input_ptr, input.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(res_ptr, input_ptr, n);

    cudaDeviceSynchronize();

    cudaMemcpy(res.data(), res_ptr, size, cudaMemcpyDeviceToHost);
    
    return res;
}