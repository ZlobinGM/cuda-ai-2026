#include "gelu_omp.h"

#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <array>
#include <cstddef>
#include <immintrin.h>

#define POW_3(x) x*x*x
#define MULT(x, y) x*y
#define SUM(x, y) x+y
#define POL(x) x*0.5f
#define sqrt_2_over_pi 0.7978845608f
#define coef 0.044715f
#define coef_2 sqrt_2_over_pi*coef

#define THRESH_BIG    5.0f
#define THRESH_SMALL  0.0f
#define C1 0.037f
#define C2 0.333f

inline float tanh_upd(float x0)
{
    if (x0 >  THRESH_BIG) return  1.0f;
    if (x0 < -THRESH_BIG) return -1.0f;

    float ex = exp(x0);
    float emx = 1./ex;
    return (ex - emx) / (ex + emx);
}

std::vector<float> GeluOMP(const std::vector<float> &input) {

    const size_t size = input.size();
    std::vector<float> output(size);

    const float* in_ptr = input.data();
    float* out_ptr = output.data();

    size_t i = 0;
    #pragma omp parallel for simd private(i)
    for (i = 0; i < size; ++i) {
        const float x = in_ptr[i];
        const float x_cube = POW_3(x);
        const float x_2 = MULT(x, x);
        const float x1 = MULT(coef_2, x_cube);
        const float x2 = MULT(sqrt_2_over_pi, x);
        const float inner = SUM(x1, x2);

        const float x_pol = POL(x);
        const float y = MULT(x_pol, tanh_upd(inner));
        out_ptr[i] = x_pol + y;
    }

    return output;
}
