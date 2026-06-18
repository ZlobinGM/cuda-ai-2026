#include "gelu_cuda.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

constexpr float SQRT_2_OVER_PI =
    std::sqrt(2.0f / static_cast<float>(M_PI));

std::vector<float> GeluRef(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        const float & x = input[i];
        output[i] = 0.5f * x *  (1.f + std::tanh(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
    }

    return output;
}


inline float fastTanh(float x) {
    return 1.f - (2.f / (1.f + std::exp(2.f * x)));
}

template <typename Func>
double Benchmark(Func&& f, int iterations = 5) {
    double best = 1e100;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        auto result = f();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        best = std::min(best, elapsed.count());

        volatile float sink = result[result.size() / 2];
        (void)sink;
    }

    return best;
}

int main() {
    constexpr size_t N = 100000000;

    std::vector<float> input(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    for (auto& x : input) {
        x = dist(rng);
    }

    std::cout << "Input size: " << N << "\n\n";

    auto ref_time = Benchmark([&]() {
        return GeluRef(input);
    });

    auto cuda_time = Benchmark([&]() {
        return GeluCUDA(input);
    });

    auto ref = GeluRef(input);
    auto cuda = GeluCUDA(input);

    auto max_error = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        max_error = std::max(
            max_error,
            std::abs(ref[i] - cuda[i]));
    }

    std::cout << "Reference : " << ref_time << " s\n";
    std::cout << "CUDA    : " << cuda_time << " s\n";
    std::cout << "Speedup   : " << ref_time / cuda_time << "x\n";
    std::cout << "Max error : " << max_error << '\n';

    return 0;
}