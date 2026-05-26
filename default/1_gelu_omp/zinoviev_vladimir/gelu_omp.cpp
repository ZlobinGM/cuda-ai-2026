#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include "gelu_omp.h"

inline float fast_exp(float x) {
    constexpr float ln_min = std::log(std::numeric_limits<float>::min());
    constexpr float ln_max = std::log(std::numeric_limits<float>::max());
    constexpr float log2e = M_LOG2E;
    constexpr float ln2 = M_LN2;
    constexpr float terms[8] = {
        1 / std::tgamma(8),
        1 / std::tgamma(7),
        1 / std::tgamma(6),
        1 / std::tgamma(5),
        1 / std::tgamma(4),
        1 / std::tgamma(3),
        1 / std::tgamma(2),
        1 / std::tgamma(1),
    };

    bool small = x < ln_min;
    x = x < ln_max ? x : ln_max;
    x = x > ln_min ? x : ln_min;

    int32_t n = static_cast<int32_t>(x * log2e + 0.5f);
    float r = x - ln2 * n;

    int32_t two_pow = (n + 126) << 23;
    float two_pow_f;
    std::memcpy(&two_pow_f, &two_pow, sizeof(two_pow_f));

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

std::vector<float> GeluOMP(const std::vector<float>& input) {
    constexpr float sqrt_2_pi = - M_2_SQRTPI * M_SQRT2;
    constexpr float sqrt_2_pi2 = sqrt_2_pi * 0.044715f;
    constexpr float sqrt_2_pi3 = sqrt_2_pi / sqrt_2_pi2;
    std::vector<float> output(input.size());
    const float* __restrict pInput = input.data();
    float* __restrict pOut = output.data();
#pragma omp parallel for simd
    for (int i = 0; i < output.size(); i += 4) {
        float in = pInput[i];
        float in1 = pInput[i + 1];
        float in2 = pInput[i + 2];
        float in3 = pInput[i + 3];
        float x = sqrt_2_pi2 * in * (in * in + sqrt_2_pi3);
        float x1 = sqrt_2_pi2 * in1 * (in1 * in1 + sqrt_2_pi3);
        float x2 = sqrt_2_pi2 * in2 * (in2 * in2 + sqrt_2_pi3);
        float x3 = sqrt_2_pi2 * in3 * (in3 * in3 + sqrt_2_pi3);
        // fast_exp
        pOut[i] = in / (1 + fast_exp(x));
        pOut[i + 1] = in1 / (1 + fast_exp(x1));
        pOut[i + 2] = in2 / (1 + fast_exp(x2));
        pOut[i + 3] = in3 / (1 + fast_exp(x3));
    }
    return output;
}
