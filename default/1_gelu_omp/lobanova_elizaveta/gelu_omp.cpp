#include "gelu_omp.h"

#include <chrono>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <omp.h>
#include <immintrin.h>

static __m256 exp256_ps(__m256 x) {
    /* Modified code. The original code is here: https://github.com/reyoung/avx_mathfun

    AVX implementation of exp
    Based on "sse_mathfun.h", by Julien Pommier
    http://gruntthepeon.free.fr/ssemath/
    Copyright (C) 2012 Giovanni Garberoglio
    Interdisciplinary Laboratory for Computational Science (LISC)
    Fondazione Bruno Kessler and University of Trento
    via Sommarive, 18
    I-38123 Trento (Italy)
    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:
    1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
     (this is the zlib license)
    */
    /* 
    To increase the compatibility across different compilers the original code is
    converted to plain AVX2 intrinsics code without ingenious macro's,
    gcc style alignment attributes etc. The modified code requires AVX2
    */
    __m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
    __m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

    __m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
    __m256   cephes_exp_C1 = _mm256_set1_ps(0.693359375);
    __m256   cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

    __m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
    __m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
    __m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
    __m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
    __m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
    __m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
    __m256   tmp           = _mm256_setzero_ps(), fx;
    __m256i  imm0;
    __m256   one           = _mm256_set1_ps(1.0f);

    x     = _mm256_min_ps(x, exp_hi);
    x     = _mm256_max_ps(x, exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
            fx    = _mm256_mul_ps(x, cephes_LOG2EF);
            fx    = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
            tmp   = _mm256_floor_ps(fx);
    __m256  mask  = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
            mask  = _mm256_and_ps(mask, one);
            fx    = _mm256_sub_ps(tmp, mask);
            tmp   = _mm256_mul_ps(fx, cephes_exp_C1);
    __m256  z     = _mm256_mul_ps(fx, cephes_exp_C2);
            x     = _mm256_sub_ps(x, tmp);
            x     = _mm256_sub_ps(x, z);
            z     = _mm256_mul_ps(x,x);

    __m256  y     = cephes_exp_p0;
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p1);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p2);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p3);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p4);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p5);
            y     = _mm256_mul_ps(y, z);
            y     = _mm256_add_ps(y, x);
            y     = _mm256_add_ps(y, one);

    /* build 2^n */
            imm0  = _mm256_cvttps_epi32(fx);
            imm0  = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
            imm0  = _mm256_slli_epi32(imm0, 23);
    __m256  pow2n = _mm256_castsi256_ps(imm0);
            y     = _mm256_mul_ps(y, pow2n);
            return y;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t inputSize = input.size();
    std::vector<float> output(inputSize);
    const float* inData = input.data();
    float* outData = output.data();

    constexpr float sqrtRes = 2 * sqrt(2.0f / M_PI);
    constexpr size_t tNum = 64;

    const __m256 vecSqrt = _mm256_set1_ps(sqrtRes);
    const __m256 vecK = _mm256_set1_ps(0.044715f);
    const __m256 vecOne = _mm256_set1_ps(1.0f);

    #pragma omp parallel for
    for (int t = 0; t < tNum; ++t) {
        const int start = t * inputSize / tNum;
        const int end = (t + 1) * inputSize / tNum;
        int i = start;
        for (; i <= end - 8; i += 8) {
            __m256 val = _mm256_loadu_ps(inData + i);
            __m256 val2 = _mm256_mul_ps(val, val);
            __m256 k = _mm256_fmadd_ps(vecK, val2, vecOne);
            __m256 tanhArg = _mm256_mul_ps(_mm256_mul_ps(vecSqrt, k), val);
            __m256 expRes = exp256_ps(tanhArg);
            __m256 tanhRes = _mm256_sub_ps(vecOne, _mm256_div_ps(vecOne, _mm256_add_ps(expRes, vecOne)));
            __m256 res = _mm256_mul_ps(val, tanhRes);
            _mm256_storeu_ps(outData + i, res);
        }
        for (; i < end; ++i) {
            const float val = inData[i];
            const float tanhArg = sqrtRes * val * (1 + 0.044715f * val * val);
            const float tanhRes = 1 - 1.0f / (std::exp(tanhArg) + 1);
            outData[i] = val * tanhRes;
        }
    }

    return output;
}

#if 0
static std::vector<float> GeluOMPRef(const std::vector<float>& input) {
    std::vector<float> output;
    output.reserve(input.size());
    for (const auto& val : input) {
        output.push_back(val * 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val))));
    }
    return output;
}

int main() {
    constexpr size_t dataSize = 33687989;
    constexpr float minVal = 0.0f;
    constexpr float maxVal = 20.0f;

    std::vector<float> input(dataSize);
    std::generate(input.begin(), input.end(), [](){
        return minVal + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (maxVal - minVal);
    });

    auto outputRef = GeluOMPRef(input);
    auto output = GeluOMP(input);
    float error = 0.0f;
    for (size_t i = 0; i < dataSize; ++i) {
        error = std::max(std::fabs(output[i] - outputRef[i]), error);
    }
    std::cout << "Absolute max error: " << error << std::endl;

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Time: " << time << " seconds" << std::endl;
}
#endif