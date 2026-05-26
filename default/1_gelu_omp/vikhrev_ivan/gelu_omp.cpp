#include "gelu_omp.h"

#include <omp.h>

#include <cmath>

constexpr float SQRT_2_DIV_PI = 0.7978845608028654f;

inline float gelu(const float x) {
    return 0.5f * x *  (1.f + std::tanh(SQRT_2_DIV_PI * (x + 0.044715f * x * x * x)));
}

std::vector<float> Gelu(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = gelu(input[i]);
    }
    return result;
}

inline float fast_tanh(float x) {
    return 1.f - (2.f / (1.f + std::exp(x * 2.f)));
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    const float* in_ptr = input.data();
    float* out_ptr = result.data();

    #pragma omp parallel for simd
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = in_ptr[i];
        const float cx3 = 0.044715f * x * x * x;
        const float tanh_val = fast_tanh(SQRT_2_DIV_PI * (x + cx3));
        out_ptr[i] = 0.5f * x *  (1.f + tanh_val);
    }
    return result;
}
