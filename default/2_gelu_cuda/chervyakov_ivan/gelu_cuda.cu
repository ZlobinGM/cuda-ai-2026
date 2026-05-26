#include "gelu_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda/std/cmath>

__global__ void geluExpSingleBufferKernel(float *inputOutput, int length)
{
    constexpr float s_FactorX3 = 0.044715f;
    constexpr float s_Half = 0.5f;
    constexpr float s_Two = 2.0f;

    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < length)
    {
        const float x = inputOutput[workIndex];
        const float x2 = x * x;
        const float z = CUDART_SQRT_2_OVER_PI_F * x * (CUDART_ONE_F + s_FactorX3 * x2);
        const float expPart = cuda::std::expf(s_Two * z) + CUDART_ONE_F;
        const float tanh = (expPart - s_Two) / expPart;
        inputOutput[workIndex] = s_Half * x * (CUDART_ONE_F + tanh);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t length = static_cast<int>(input.size());
    size_t allocLength = length * sizeof(float);
    std::vector<float> output(length);

    float *hostA = const_cast<float *>(input.data());
    float *hostB = const_cast<float *>(output.data());

    float *devA = nullptr;

    cudaMalloc(&devA, allocLength);

    cudaMemcpy(devA, hostA, allocLength, cudaMemcpyHostToDevice);

    int blockSize, minGridSize, blocks;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)geluExpOptSingleBufferKernel, 0, length);
    blocks = (length + blockSize - 1) / blockSize;
    geluExpSingleBufferKernel<<<blocks, blockSize>>>(devA, length);

    cudaDeviceSynchronize();

    cudaMemcpy(hostB, devA, allocLength, cudaMemcpyDeviceToHost);

    cudaFree(devA);

    return output;
}