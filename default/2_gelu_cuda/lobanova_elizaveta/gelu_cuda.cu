#include "gelu_cuda.h"

#include <cuda/cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>

__global__ void GeluKernel(float* input, int inputSize) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < inputSize) {
        float val = input[workIndex];
        input[workIndex] = val * (1 - 1.0f / (expf(1.59576912f * val * (1 + 0.044715f * val * val)) + 1));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int inputSize = input.size();
    std::vector<float> output(inputSize);
    const float* inData = input.data();
    float* outData = output.data();

    float* devInput = nullptr;
    cudaMalloc(&devInput, inputSize * sizeof(float));
    cudaMemcpy(devInput, inData, inputSize * sizeof(float), cudaMemcpyDefault);

    int threads = 256;
    int blocks = cuda::ceil_div(inputSize, threads);
    GeluKernel<<<blocks, threads>>>(devInput, inputSize);
    cudaDeviceSynchronize();

    cudaMemcpy(outData, devInput, inputSize * sizeof(float), cudaMemcpyDefault);
    cudaFree(devInput);

    return output;
}

#if 0
static std::vector<float> GeluRef(const std::vector<float>& input) {
    std::vector<float> output;
    output.reserve(input.size());
    for (const auto& val : input) {
        output.push_back(val * 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val))));
    }
    return output;
}

int main() {
    constexpr size_t dataSize = 33687989;
    constexpr float minVal = 0.0f;
    constexpr float maxVal = 20.0f;

    std::vector<float> input(dataSize);
    std::generate(input.begin(), input.end(), [](){
        return minVal + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (maxVal - minVal);
    });

    auto outputRef = GeluRef(input);
    auto output = GeluCUDA(input);
    float error = 0.0f;
    for (size_t i = 0; i < dataSize; ++i) {
        error = std::max(std::fabs(output[i] - outputRef[i]), error);
    }
    std::cout << "Absolute max error: " << error << std::endl;

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluCUDA(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Time: " << time << " seconds" << std::endl;
}
#endif