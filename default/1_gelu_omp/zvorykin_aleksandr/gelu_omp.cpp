#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

// Anonymous namespace
namespace
{
    // Used constants
    constexpr float s_TwoSqrt2OverPi = 1.595769f;
    constexpr float s_Sqrt2OverPi    = 0.797884f;
    constexpr float s_ScaleX3        = 0.044715f;
    constexpr float s_Half           = 0.5f;
    constexpr float s_One            = 1.0f;
    constexpr float s_Two            = 2.0f;

#pragma omp declare simd notinbranch
    // Base tanh GELU implementation
    inline float geluTanhImpl(float x) noexcept
    {
        const float x2  = x * x;
        const float arg = s_Sqrt2OverPi * x * (s_One + s_ScaleX3 * x2);
        return s_Half * x * (s_One + std::tanh(arg));
    }

#pragma omp declare simd notinbranch
    // Exp replaced GELU implementation
    inline float geluExpImpl(float x) noexcept
    {
        const float x2      = x * x;
        const float arg     = s_TwoSqrt2OverPi * x * (s_One + s_ScaleX3 * x2);
        const float expOne  = s_One + std::exp(arg);
        const float tanhRes = (expOne - s_Two) / expOne;
        return s_Half * x * (s_One + tanhRes);
    }
}

std::vector<float> GeluOMP(const std::vector<float>& input) 
{
    // Place your implementation here
    const size_t size = input.size();
    std::vector<float> result(size);

    float *__restrict p_out = result.data();
    const float *__restrict p_in = input.data();

#pragma omp parallel for simd schedule(static) aligned(p_in, p_out : 32)
    for (size_t j = 0; j < size; ++j)
    {
        p_out[j] = geluExpImpl(p_in[j]);
    }

    return result;
}
