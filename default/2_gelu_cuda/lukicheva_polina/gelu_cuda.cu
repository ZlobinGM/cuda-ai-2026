#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

#include "gelu_cuda.h"

__global__ void GeluKernel(float* input, float* output, int nElems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElems) {

        float el = input[idx];
        output[idx] = 0.5f * el * (1 + tanhf(0.79788456f * el * (1.f + 0.044715f * el * el)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int nElems = input.size();
    std::vector<float> output(nElems);

    const float* hInput = input.data();
    float* hOutput = output.data();

    float* dInput = nullptr;
    float* dOutput = nullptr;
    cudaMalloc(&dInput, nElems*sizeof(float)); 
    cudaMalloc(&dOutput, nElems*sizeof(float));

    cudaMemcpy(dInput, hInput, nElems*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, hOutput, nElems*sizeof(float), cudaMemcpyHostToDevice);

    const int block_size = 256;
    int num_blocks = (nElems + block_size - 1) / block_size;

    GeluKernel <<< num_blocks, block_size >>> (dInput, dOutput, nElems);

    cudaMemcpy(hOutput, dOutput, nElems * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dInput);
    cudaFree(dOutput);
    
    return output;

}

#if 0
std::vector<float> GeluRef(const std::vector<float>& input) {
    size_t nElems = input.size();
    std::vector<float> output(nElems);

    constexpr float CONST_VALUE = 0.044715f;
    constexpr float SQRT_2_PI = 0.79788456f;

    for (size_t i = 0; i < nElems; ++i) {
        float el = input[i];
        output[i] = 0.5f * el * (1 + std::tanh(SQRT_2_PI * el * (1.f + CONST_VALUE * el * el)));
    }

    return output;
}


int main() {
    size_t nElems = 100000000u;
    std::vector<float> input(nElems);
    for (size_t i = 0; i < nElems; ++i) {
        input[i] = ((float)rand() / RAND_MAX) * 20.f - 10.f;
    }

    auto ref_res = GeluRef(input);
    auto omp_res = GeluCUDA(input);

    float error = 0.0f;
    for (size_t i = 0; i < nElems; ++i) {
        error = std::max(std::abs(ref_res[i] - omp_res[i]), error);
    }
    std::cout << "Absolute max error: " << error << std::endl;

    int nIters = 10;
    double min_t = 0.f;

    for (int i = 0; i < nIters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        omp_res = GeluCUDA(input);
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        double t = duration.count();
        min_t = i == 0 ? t : std::min(min_t, t);
    }

    std::cout << "Min execution time: \t" << min_t << std::endl;
}
#endif
