#include "gelu_cuda.h"

#include "gelu_cuda.h"
#include <vector>

#define C0 0.7978845608028654f
#define C1 0.044715f
#define THRESH_BIG    5.0f
#define BLOCK_SIZE 256

#define GET_NUM_BLOCK(n) (n+BLOCK_SIZE-1)/BLOCK_SIZE

__global__ void gelu_tanh_fast(float* x, float* y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x3 = x[idx] * x[idx] * x[idx];
        float inner = C0 * (x[idx] + C1 * x3);

        float ex = exp(inner);
        float emx = 1./ex;
        float tanh_inner = (ex - emx) / (ex + emx);

        //y[idx] = (x[idx] >  THRESH_BIG) ? x[idx] : (x[idx] < THRESH_BIG) ? 0.0f : 0.5f * x[idx] * (1.0f + tanh_inner);
        y[idx] = 0.5f * x[idx] * (1.0f + tanh_inner);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    int N = static_cast<int>(input.size());
    int numBlock = GET_NUM_BLOCK(N);

    const float* in = input.data();

    std::vector<float> output(N, 0.0f);
    float* out = output.data();

    float *dInput = nullptr;
    cudaMalloc(&dInput, N * sizeof(float));

    float* dOutput = nullptr;
    cudaMalloc(&dOutput, N*sizeof(float));

    cudaMemcpy(dInput, in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, out, N*sizeof(float), cudaMemcpyHostToDevice);

    gelu_tanh_fast<<<numBlock, BLOCK_SIZE>>>(dInput, dOutput, N);

    cudaMemcpy(out, dOutput, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dInput);
    cudaFree(dOutput);
    
    return output;
}