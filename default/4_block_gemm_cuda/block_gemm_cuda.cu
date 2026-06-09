#include "block_gemm_cuda.h"

__global__ void blockGemmKernelExtShMem(const float *a, const float *b, float *c, int n, int blockSize)
{
    extern __shared__ float shared_mem[];
    float* As = shared_mem; 
    float* Bs = &shared_mem[blockSize * blockSize];

    int irow = blockIdx.y * blockSize + threadIdx.y;
    int icol = blockIdx.x * blockSize + threadIdx.x;

    float zero = 0.0f;
    float sum = 0.0f;
    int numTiles = (n + blockSize - 1) / blockSize;

    for (int t = 0; t < numTiles; ++t)
    {
        int aCol = t * blockSize + threadIdx.x;
        int bRow = t * blockSize + threadIdx.y;

        As[threadIdx.y * blockSize + threadIdx.x] = (irow < n && aCol < n) ? a[irow * n + aCol] : zero;
        Bs[threadIdx.y * blockSize + threadIdx.x] = (bRow < n && icol < n) ? b[bRow * n + icol] : zero;

        __syncthreads();

        for (int k = 0; k < blockSize; ++k)
        {
            sum += As[threadIdx.y * blockSize + k] * Bs[k * blockSize + threadIdx.x];
        }

        __syncthreads();
    }

    if (irow < n && icol < n)
    {
        c[irow * n + icol] = sum;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int deviceId = 0; 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    // 1. Threads upper bound: threads_per_block = BLOCK_SIZE * BLOCK_SIZE <= maxThreadsPerBlock
    int max_block_side_threads = std::sqrt(prop.maxThreadsPerBlock);
    
    // 2. Shared Memory upper bound : 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float) <= sharedMemPerBlock
    int max_shared_bytes = prop.sharedMemPerBlock;
    int max_block_side_shared = std::sqrt(max_shared_bytes / (2 * sizeof(float)));

    int calculated_block_size = std::min({max_block_side_threads, max_block_side_shared, prop.maxThreadsDim[0]});
    
    //3. Match with warp size
    if (calculated_block_size >= 32) {
        calculated_block_size = 32;
    } else if (calculated_block_size >= 16) {
        calculated_block_size = 16;
    } else {
        calculated_block_size = 8;
    }

    size_t sharedMemBytes = 2 * calculated_block_size * calculated_block_size * sizeof(float);


    size_t N = n * n;
    size_t size = N * sizeof(float);
    std::vector<float> c(N);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    float *host_a = const_cast<float *>(a.data());
    float *host_b = const_cast<float *>(b.data());
    float *host_c = const_cast<float *>(c.data());

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_a)), size, cudaHostRegisterDefault);
    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_b)), size, cudaHostRegisterDefault);
    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_c)), size, cudaHostRegisterDefault);

    cudaMemcpyAsync(dev_a, host_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_b, host_b, size, cudaMemcpyHostToDevice, stream);

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)naiveGemmKernel, 0, N);

    dim3 threads(calculated_block_size, calculated_block_size);
    int blocksNum = cuda::ceil_div(n, calculated_block_size);
    dim3 blocks(blocksNum, blocksNum);
    blockGemmKernelExtShMem<<<blocks, threads, sharedMemBytes, stream>>>(dev_a, dev_b, dev_c, n, calculated_block_size);

    cudaMemcpyAsync(host_c, dev_c, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return c;
}