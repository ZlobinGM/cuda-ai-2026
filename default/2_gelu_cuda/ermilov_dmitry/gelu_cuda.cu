#include "gelu_cuda.h"

#include <cstring>
#include <cuda_runtime.h>

__device__  __forceinline__ float fast_tanh(float x) {
    return 1.f - (2.f / (1.f + __expf(2.f * x)));
}
constexpr float SQRT_2_OVER_PI  = 0.7978845608028654f;

__global__ void GeluKernel(float* in_out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = in_out[idx];
        float t = fast_tanh(SQRT_2_OVER_PI  * (x + 0.044715f * x * x * x));
        in_out[idx] = 0.5f * x *  (1.f + t);
    }
}

constexpr size_t THREADS = 256;

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();

    if (n == 0)
        return {};
    float* v_buffer = nullptr;

    cudaMalloc(&v_buffer, n * sizeof(float));
    cudaMemcpyAsync(
        v_buffer,
        input.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice);

    auto blocks = static_cast<size_t>((n + THREADS - 1) / THREADS);

    GeluKernel<<<blocks, THREADS>>>(v_buffer, n);

    std::vector<float> output(n);

    cudaMemcpyAsync(
        output.data(),
        v_buffer,
        n * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(v_buffer);
    return output;
}