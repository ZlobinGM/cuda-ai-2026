#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iostream>

#pragma GCC optimize("O3")

#ifdef __x86_64__
#pragma GCC target("avx2,fma")
#include <immintrin.h>
#define CASE 5
#else
#define CASE 3
#endif

#if CASE == 5
//!< Using Lambert’s continued fraction (7th-degree approximation) and applying Horner’s scheme
inline __m256 avx2_tanh(__m256 x) {
    
    __m256 xs = _mm256_mul_ps(x, x);
    __m256 n = _mm256_add_ps(xs, _mm256_set1_ps(378));
    n = _mm256_mul_ps(n, xs);
    n = _mm256_add_ps(n, _mm256_set1_ps(17325));
    n = _mm256_mul_ps(n, xs);
    n = _mm256_add_ps(n, _mm256_set1_ps(135135));
    n = _mm256_mul_ps(n, x);

    __m256 d = _mm256_mul_ps(xs, _mm256_set1_ps(28));
    d = _mm256_add_ps(d, _mm256_set1_ps(3150));
    d = _mm256_mul_ps(d, xs);
    d = _mm256_add_ps(d, _mm256_set1_ps(62370));
    d = _mm256_mul_ps(d, xs);
    d = _mm256_add_ps(d, _mm256_set1_ps(135135));

    n = _mm256_div_ps(n, d);
    n = _mm256_min_ps(n, _mm256_set1_ps(1));
    n = _mm256_max_ps(n, _mm256_set1_ps(-1));
    return n;
}
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
#if CASE == 0
    // Reference implementation
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 0.5f * input[i] * (1.f + tanh(sqrt(2 / M_PI) * (input[i] + 0.044715f * pow(input[i], 3))));
    }
#elif CASE == 1
    // Custom tanh function
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 0.5f * input[i] * (1.f + my_tanh(sqrt(2.f / M_PI) * (input[i] + 0.044715f * pow(input[i], 3))));
    }
#elif CASE == 2
    // OpenMP loop vectorization
    const float coeff1 = 0.5f;
    const float coeff2 = std::sqrt(2.0f / M_PI);
    const float coeff3 = 0.044715f;

    #pragma omp simd
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = coeff2 * (x + coeff3 * x3);
        output[i] = coeff1 * x * (1.0f + my_tanh(inner));
    }
#elif CASE == 3
    // OpenMP loop unrolling
    const float coeff1 = 0.5f;
    const float coeff2 = std::sqrt(2.0f / M_PI);
    const float coeff3 = 0.044715f;

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = coeff2 * (x + coeff3 * x3);
        output[i] = coeff1 * x * (1.0f + my_tanh(inner));
    }
#elif CASE == 4
    // OpenMP loop unrolling - using multiple threads
    const float coeff1 = 0.5f;
    const float coeff2 = std::sqrt(2.0f / M_PI);
    const float coeff3 = 0.044715f;

    #pragma omp parallel num_threads(8)
    #pragma omp for
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = coeff2 * (x + coeff3 * x3);
        output[i] = coeff1 * x * (1.0f + my_tanh(inner));
    }

#elif CASE == 5
    // with AVX2 intrinsics - build with: g++ -O3 -mavx2 -mfma -std=c++11
    size_t i = 0;
    
    const __m256 coeff1 = _mm256_set1_ps(0.5f);
    const __m256 coeff2 = _mm256_set1_ps(std::sqrt(2.0f / M_PI));
    const __m256 coeff3 = _mm256_set1_ps(0.044715f);
    const __m256 one = _mm256_set1_ps(1.0f);
    
    for (; i + 8 <= input.size(); i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        
        __m256 inner = _mm256_fmadd_ps(coeff3, x3, x);  // x + coeff3 * x^3
        inner = _mm256_mul_ps(coeff2, inner);
        
        __m256 tanh_val = avx2_tanh(inner);
        
        __m256 sum = _mm256_add_ps(one, tanh_val);
        __m256 result = _mm256_mul_ps(coeff1, _mm256_mul_ps(x, sum));
        
        _mm256_storeu_ps(&output[i], result);
    }
    
    for (; i < input.size(); i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + my_tanh(std::sqrt(2.0f / M_PI) * 
                                              (x + 0.044715f * x * x * x)));
    }
#endif
    return output;
}

// int main() {
//     // ...
//     std::size_t size = 134217728;
//     std::vector<float> input(size);
//     float left = -20.0;
//     float right = -left;
//     float step = (right - left) / size;
//     for (int i = 0; i < size; ++i){
//         input[i] = left + step * static_cast<float>(i);
//     }

//     // Warming-up
//     GeluOMP(input);

//     // Performance Measuring
//     std::vector<double> time_list;
//     for (int i = 0; i < 20; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         auto res = GeluOMP(input);
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> duration = end - start;
//         time_list.push_back(duration.count());
//     }
//     double time = *std::min_element(time_list.begin(), time_list.end());

//     std::cout << "Time: " << time << "s" << std::endl;
// }