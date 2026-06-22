#include "naive_gemm_cuda.h"

#include <cuda/cmath>

__global__ void NaiveGemmCUDAImpl(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int size = a.size();
    const int bSize = size * sizeof(float);

    float* cHostPtr = nullptr;
    float* devicePtr = nullptr;

    cudaMalloc(&devicePtr, 3 * bSize);
    float* aDevicePtr = devicePtr;
    float* bDevicePtr = devicePtr + size;
    float* cDevicePtr = devicePtr + 2 * size;

    cudaMemcpy(aDevicePtr, a.data(), bSize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevicePtr, b.data(), bSize, cudaMemcpyHostToDevice);
    cudaMemset(cDevicePtr, 0, bSize);

    constexpr int nThreads = 16;
    int blocks = cuda::ceil_div(n, nThreads);
    dim3 threadsDim(nThreads, nThreads);
    dim3 blocksDim(blocks, blocks);
    NaiveGemmCUDAImpl<<<blocksDim, threadsDim>>>(aDevicePtr, bDevicePtr, cDevicePtr, n);

    std::vector<float> c(size);
    cHostPtr = c.data();

    cudaDeviceSynchronize();
    cudaMemcpy(cHostPtr, cDevicePtr, bSize, cudaMemcpyDeviceToHost);
    cudaFree(devicePtr);

    return c;
}
