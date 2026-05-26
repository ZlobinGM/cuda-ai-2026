#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cuda/cmath>

// Anonymous namespace
namespace
{
    // Used constants
    constexpr float s_TwoSqrt2OverPi = 1.595769f;
    constexpr float s_ScaleX3        = 0.044715f;
    constexpr float s_One            = 1.0f;

    __global__ void geluSigmoidImpl(float * __restrict__ data, int length)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < length)
        {
            const float x      = data[index];
            const float x2     = x * x;
            const float expOne = cuda::std::expf(s_TwoSqrt2OverPi * x * (s_One + s_ScaleX3 * x2)) + s_One;
            data[index] = x - x / expOne;
        }
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    // Place your implementation here
    const size_t size = static_cast<int>(input.size());
    const size_t bitSize = size * sizeof(float);
    std::vector<float> output(size);

    float *deviceData = nullptr;
    cudaMalloc(&deviceData, bitSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(deviceData, input.data(), bitSize, cudaMemcpyHostToDevice, stream);

    constexpr int numThreads = 256;
    int numBlocks = (static_cast<int>(size) + numThreads - 1) / numThreads;
    geluSigmoidImpl<<<numBlocks, numThreads, 0, stream>>>(deviceData, static_cast<int>(size));
  
    cudaMemcpyAsync(output.data(), deviceData, bitSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(deviceData);
    cudaStreamDestroy(stream);

    return output;
}
