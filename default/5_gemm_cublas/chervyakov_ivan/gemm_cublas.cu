
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>
#include <vector>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t N = n * n;
    size_t bytes = N * sizeof(float);
    std::vector<float> c(N);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev_b, n, dev_a, n, &beta, dev_c, n);
   
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cublasDestroy(cublasHandle); 

    return c;
}