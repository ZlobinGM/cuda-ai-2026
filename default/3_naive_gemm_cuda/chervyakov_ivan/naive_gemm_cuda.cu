#include "naive_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>

__global__ void naiveGemmKernel(const float *a, const float *b, float *c, int n)
{

    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;

    if (irow < n && icol < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += a[irow * n + k] * b[k * n + icol];
        }
        c[irow * n + icol] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t N = n * n;
    size_t size = N * sizeof(float);
    std::vector<float> c(N);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), size, cudaMemcpyHostToDevice);

    constexpr int tileSize = 16;

    dim3 blockDims(tileSize, tileSize);

    int sizeGridDim = cuda::ceil_div(n, tileSize);
    dim3 gidDims(sizeGridDim, sizeGridDim);

    naiveGemmKernel<<<gidDims, blockDims>>>(dev_a, dev_b, dev_c, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), dev_c, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return c;
}