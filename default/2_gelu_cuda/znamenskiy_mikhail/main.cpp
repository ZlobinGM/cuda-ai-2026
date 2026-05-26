#include <gelu_cuda.h>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

namespace {
    std::vector<float> GeluRef(const std::vector<float>& input) {
        std::vector<float> result(input.size());
        for (size_t index = 0; index < input.size(); ++index) {
            float x = input[index];
            result[index] = 0.5 * x * (1 + std::tanh(std::sqrt(2.f/M_PI)*(x+0.044715f*std::pow(x,3))));
        }
        return result;
    }
}

int main() {

constexpr size_t ARRAY_SIZE = 134217728;

std::vector<float> inputData(ARRAY_SIZE);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(-100.f, 100.f);
std::cout << "Generating random data" << std::endl;
for (size_t n = 0; n < ARRAY_SIZE; ++n)
    inputData[n] = dis(gen);
std::cout << "Generating DONE" << std::endl;

std::cout << "Ref calculations" << std::endl;
std::chrono::steady_clock::time_point beginRef = std::chrono::steady_clock::now();
auto resultRef = GeluRef(inputData);
std::chrono::steady_clock::time_point endRef = std::chrono::steady_clock::now();
std::cout << "Ref calculations DONE" << std::endl;
std::cout << "Time REF = " << std::chrono::duration_cast<std::chrono::milliseconds>(endRef - beginRef).count() << "[ms]" << std::endl;

std::cout << "Warming up" << std::endl;
GeluCUDA(inputData);
std::cout << "Warming up DONE" << std::endl;

std::cout << "Measurements" << std::endl;
std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
auto result = GeluCUDA(inputData);
std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
std::cout << "Measurements done" << std::endl;
std::cout << "Time OPT = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

std::cout << "Accuracy check" << std::endl;
float error = 0.0f;
for (size_t n = 0; n < ARRAY_SIZE; ++n) {
    error = std::max(std::abs(result[n] - resultRef[n]), error);
    if (std::isnan(error)) {
        std::cout << "NAN error - index = " << n << " result = " << result[n] << " ref = " << resultRef[n] << std::endl;
        return 1;
    }
}
std::cout << "Accuracy check FINISHED" << std::endl;
std::cout << "Max error = " << error << std::endl;

return 0;
}
