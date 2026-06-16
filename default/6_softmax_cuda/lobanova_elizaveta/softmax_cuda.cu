#include "softmax_cuda.h"

#include <cuda/cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <float.h>
#include <thread>

#define BLOCK_SIZE 256

__global__ void SoftmaxCUDAKernel(float* input, int row_count, int row_size) {
    int row_ix = blockIdx.x;
    if (row_ix >= row_count) {
        return;
    }

    int t_id = threadIdx.x;
    float* row_input_data = input + row_ix * row_size;
    float t_max = -INFINITY;
    for (int col = t_id; col < row_size; col += blockDim.x) {
        t_max = fmaxf(t_max, row_input_data[col]);
    }

    extern __shared__ float shared_mem[];
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        t_max = fmaxf(t_max, __shfl_down_sync(0xFFFFFFFF, t_max, offset));
    }

    if (t_id == 0) {
        shared_mem[t_id] = t_max;
    }
     __syncthreads();

    float global_max = shared_mem[0];

    float t_sum = 0.0f;
    for (int col = t_id; col < row_size; col += blockDim.x) {
        float exp_val = expf(row_input_data[col] - global_max);
        row_input_data[col] = exp_val;
        t_sum += exp_val;
    }

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        t_sum += __shfl_down_sync(0xFFFFFFFF, t_sum, offset);
    }
    if (t_id == 0) {
        shared_mem[t_id] = t_sum;
    }
    __syncthreads();

    float k = 1.0f / shared_mem[0];
    for (int col = t_id; col < row_size; col += blockDim.x) {
        row_input_data[col] *= k;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const int data_size = input.size();
    std::vector<float> output;
    std::thread t([&output, data_size](){
        output.resize(data_size);
    });
    const int row_size = data_size / row_count;
    const float* input_data = input.data();

    float* devInput = nullptr;
    cudaMalloc(&devInput, data_size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(devInput, input_data, data_size * sizeof(float), cudaMemcpyHostToDevice, stream);

    SoftmaxCUDAKernel<<<row_count, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(devInput, row_count, row_size);

    t.join();
    cudaMemcpyAsync(output.data(), devInput, data_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(devInput);
    return output;
}

#if 0
std::vector<float> SoftmaxRef(const std::vector<float>& input, int row_count) {
    int input_size = input.size();
    int row_size = input_size / row_count;
    std::vector<float> output(input_size);
    for (int i = 0; i < row_count; ++i) {
        float max_val = *std::max_element(input.begin() + i * row_size, input.begin() + i * (row_size + 1));
        float sum = 0.0f;
        std::vector<float> exponents(row_size);
        for (int j = 0; j < row_size; ++j) {
            exponents[j] = std::exp(input[i * row_size + j] - max_val);
            sum += exponents[j];
        }
        float k = 1.0f / sum;
        for (int j = 0; j < row_size; ++j) {
            output[i * row_size + j] = exponents[j] * k;
        }
    }
    return output;
}

int main() {
    constexpr size_t row_count = 8192;
    constexpr size_t row_size = 16384;
    constexpr float minVal = -10.0f;
    constexpr float maxVal = 10.0f;

    std::vector<float> a(row_count * row_size);
    std::generate(a.begin(), a.end(), [](){
        return minVal + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (maxVal - minVal);
    });

    auto cRef = SoftmaxRef(a, row_count);
    auto c = SoftmaxCUDA(a, row_count);
    float error = 0.0f;
    for (size_t i = 0; i < row_count * row_size; ++i) {
        error = std::max(std::fabs(c[i] - cRef[i]), error);
    }
    std::cout << "Max error: " << error / maxVal << std::endl;

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        SoftmaxCUDA(a, row_count);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Time: " << time << " seconds" << std::endl;
}
#endif