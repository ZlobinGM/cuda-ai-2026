#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input);

inline float my_tanh(float x) {
    float e_x2 = exp(2.f*x);
    return 1.f - 2.f/(e_x2+1.f);
}

#endif // __GELU_OMP_H