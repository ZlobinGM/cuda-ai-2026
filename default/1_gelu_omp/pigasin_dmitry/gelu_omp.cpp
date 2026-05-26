#include "gelu_omp.h"

#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float> &input) {
    const size_t size = input.size();
    std::vector<float> output(size);

    const float *__restrict in_ptr = input.data();
    float *__restrict out_ptr = output.data();

    // Pad approximant: tanh(w) = w*(945 + 105v + v^2) / (945 + 420v + 15v^2)
    // Where v = w^2, so P = 945 + 105v + v^2, Q = 945 + 420v + 15v^2
    const float c = 0.7978845608f;       // sqrt(2/pi)
    const float k = 0.044715f;           // GELU coefficient

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        const float x = in_ptr[i];
        const float x2 = x * x;
        const float inner = c * x * (1.0f + k * x2);
        const float v = inner * inner;
        const float P = 945.0f + v * (105.0f + v);
        const float Q = 945.0f + v * (420.0f + 15.0f * v);
        out_ptr[i] = 0.5f * x * (1.0f + inner * P / Q);
    }

    return output;
}
