#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "gelu_cuda.h"

__device__ float fast_exp(float x) {
    static const float ln_min = -87.3365f;
    static const float ln_max = 88.7228f;
    constexpr float log2e = M_LOG2E;
    constexpr float ln2 = M_LN2;
    static const float terms[8] = {
        0.000198413f,
        0.00138889f,
        0.00833333f,
        0.0416667f,
        0.166667f,
        0.5f,
        1.f,
        1.f
    };

    bool small = x < ln_min;
    x = x < ln_max ? x : ln_max;
    x = x > ln_min ? x : ln_min;

    int32_t n = static_cast<int32_t>(x * log2e + 0.5f);
    float r = x - ln2 * n;

    int32_t two_pow = (n + 126) << 23;
    float two_pow_f = __int_as_float(two_pow);
    // std::memcpy(&two_pow_f, &two_pow, sizeof(two_pow_f));

    float exp_r = terms[0];
    for (int i = 1; i < 8; ++i) {
        exp_r = exp_r * r + terms[i];
    }

    float exp = exp_r * two_pow_f;
    exp *= 2.f;
    if (small) {
        exp = 0.f;
    }
    return exp;
}

__global__ void GeluCUDA(const float* in, float* out, int n) {
    static constexpr float sqrt_2_pi2 = -0.0713548138737678528F;
    static constexpr float sqrt_2_pi3 = 22.363861083984375F;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = in[i];
        float x = sqrt_2_pi2 * val * (val * val + sqrt_2_pi3);
        out[i] = val / (1 + fast_exp(x));
        // out[i] = 0.5f * x * (1 + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    std::vector<float> output(size);
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    float *d_in, *d_out; 
    cudaMalloc(&d_in, size*sizeof(float)); 
    cudaMalloc(&d_out, size*sizeof(float));
    cudaMemcpy(d_in, input.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    GeluCUDA<<<num_blocks, block_size>>>(d_in, d_out, size);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_out, size*sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}