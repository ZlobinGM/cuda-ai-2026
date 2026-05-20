#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

#include "gelu_omp.h"

double fast_tanh(double x) {
    double e2x = std::exp(2.f * x);
    return (e2x - 1.f) / (e2x + 1.f);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t nElems = input.size();
    std::vector<float> output(nElems);

    constexpr float CONST_V = 0.044715f;
    constexpr float SQRT_2_PI = std::sqrt(2.f / M_PI);

    #pragma omp parallel for
    for (size_t i = 0; i < nElems; ++i) {
        float el = input[i];
        output[i] = 0.5f * el * (1 + fast_tanh(SQRT_2_PI * el * (1.f +  CONST_V * el * el)));
    }

    return output;
}

std::vector<float> GeluRef(const std::vector<float>& input) {
    size_t nElems = input.size();
    std::vector<float> output(nElems);

    constexpr float CONST_VALUE = 0.044715f;
    constexpr float SQRT_2_PI = std::sqrt(2.f / M_PI);

    for (size_t i = 0; i < nElems; ++i) {
        float el = input[i];
        output[i] = 0.5f * el * (1 + std::tanh(SQRT_2_PI * el * (1.f + CONST_VALUE * el * el)));
    }

    return output;
}

#if 0
int main() {
    size_t nElems = 100000000u;
    std::vector<float> input(nElems);
    for (size_t i = 0; i < nElems; ++i) {
        input[i] = ((float)rand() / RAND_MAX) * 20.f - 10.f;
    }

    auto ref_res = GeluRef(input);
    auto omp_res = GeluOMP(input);

    float error = 0.0f;
    for (size_t i = 0; i < nElems; ++i) {
        error = std::max(std::abs(ref_res[i] - omp_res[i]), error);
    }
    std::cout << "Absolute max error: " << error << std::endl;

    int nIters = 10;
    double min_t = 0.f;

    for (int i = 0; i < nIters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        omp_res = GeluOMP(input);
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        double t = duration.count();
        min_t = i == 0 ? t : std::min(min_t, t);
    }

    std::cout << "Min execution time: \t" << min_t << std::endl;
}
#endif
