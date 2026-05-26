#include "gelu_cuda.h"

#include <cuda/cmath>

__global__ void GeluCUDAImpl(float* gpuBuffer, int bufferSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < bufferSize) {
        // Reusing the same buffer for in/out data
        float x = gpuBuffer[index];
        gpuBuffer[index] = x - x / (cuda::std::expf(1.59576912f * x * (1.f + 0.044715f * x * x)) + 1.f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int inputSize = input.size();
    const int inputSizeBytes = inputSize * sizeof(float);
    const float* inputData = input.data();

    float* gpuBuffer = nullptr;
    cudaMalloc(&gpuBuffer, inputSizeBytes);
    cudaMemcpy(gpuBuffer, inputData, inputSizeBytes, cudaMemcpyDefault);

    constexpr int threads = 256;
    int blocks = cuda::ceil_div(inputSize, threads);
    GeluCUDAImpl<<<blocks, threads>>>(gpuBuffer, inputSize);

    // Allocating result buffer while CUDA calculations are running
    std::vector<float> result(inputSize);
    float* resultData = result.data();

    cudaDeviceSynchronize();
    cudaMemcpy(resultData, gpuBuffer, inputSizeBytes, cudaMemcpyDefault);
    cudaFree(gpuBuffer);

    return result;
}
