#include "gelu_omp.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <omp.h>
#include <chrono>
#include <cmath>

namespace
{
    constexpr float s_Sqrt2OverPi = 0.7978845608028654f;
    constexpr float s_FactorX3 = 0.044715f;
    constexpr float s_Half = 0.5f;
    constexpr float s_One = 1.0f;
    constexpr float s_Two = 2.0f;

#pragma omp declare simd notinbranch
    inline float geluExpOpt(float x) noexcept
    {
        const float x2 = x * x;
        const float z = s_Sqrt2OverPi * x * (s_One + s_FactorX3 * x2);
        const float expPart = std::exp(s_Two * z) + s_One;
        const float tanh = (expPart - s_Two) / expPart;
        return s_Half * x * (s_One + tanh);
    }

/*#pragma omp declare simd notinbranch
    inline float geluTanh(float x) noexcept
    {
        const float x2 = x * x;
        const float inner = s_Sqrt2OverPi * x * (s_One + s_FactorX3 * x2);
        return s_Half * x * (s_One + std::tanh(inner));
    }
*/
}

/*std::vector<float> geluTanhBase(const std::vector<float> &input)
{
    const size_t n = input.size();
    std::vector<float> result(n);

    float *__restrict p_out = result.data();
    const float *__restrict p_in = input.data();

    for (size_t j = 0; j < n; ++j)
    {
        p_out[j] = geluTanh(p_in[j]);
    }

    return result;
}*/

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    const size_t n = input.size();
    std::vector<float> result(n);

    float *__restrict p_out = result.data();
    const float *__restrict p_in = input.data();

#pragma omp parallel for simd schedule(static) aligned(p_in, p_out : 32)
    for (size_t j = 0; j < n; ++j)
    {
        p_out[j] = geluExpOpt(p_in[j]);
    }

    return result;
}

/*
inline std::vector<float> generateTestData(size_t size, float min_val = -5.0f, float max_val = 5.0f)
{
    std::vector<float> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> urod(min_val, max_val);
    for (auto &x : data)
    {
        x = urod(gen);
    }
    return data;
}

int main()
{
    auto *GeluOmpRef = &geluTanhBase;
    auto *GeluOmp = &GeluOMP;
 
    auto input = generateTestData(1000000);

    auto out = GeluOmp(input);
    auto refOut = GeluOmpRef(input);
    float maxDiff = 0;
    for (size_t i = 0; i < out.size(); i++)
        maxDiff = std::max(maxDiff, std::fabs(refOut[i] - out[i]));
    printf("Max difference::%f\n", maxDiff);

    std::vector<float> g_out;
    std::vector<double> time_list;
    for (size_t i = 0; i < 1000; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto outLocal = GeluOmp(input);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        g_out = std::move(outLocal);
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("%f\n", time);

    return 0;
}*/