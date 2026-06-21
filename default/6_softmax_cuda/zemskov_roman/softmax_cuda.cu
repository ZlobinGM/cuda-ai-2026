#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>

constexpr int WARP_SIZE = 32;
constexpr int VECTOR_SIZE = 4; 

__global__ void softmax_kernel_optimized_inplace(
    float* __restrict__ data,
    int row_count,
    int row_size
) {
    extern __shared__ float shared_mem[];
    float* row_max_shared = shared_mem;
    float* row_sum_shared = &shared_mem[blockDim.x / WARP_SIZE];
    
    int row_idx = blockIdx.x;
    if (row_idx >= row_count) return;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    float* row_data = data + row_idx * row_size;
    
    float local_max = -INFINITY;
    
    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        float4 vals = reinterpret_cast<const float4*>(row_data + i)[0];
        local_max = fmaxf(local_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (lane == 0) {
        row_max_shared[warp_id] = local_max;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane < num_warps) ? row_max_shared[lane] : -INFINITY;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) {
            row_max_shared[0] = val;
        }
    }
    __syncthreads();
    float row_max = row_max_shared[0];
    
    float local_sum = 0.0f;
    
    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        float4 vals = reinterpret_cast<const float4*>(row_data + i)[0];
        float4 exp_vals;
        
        exp_vals.x = expf(vals.x - row_max);
        exp_vals.y = expf(vals.y - row_max);
        exp_vals.z = expf(vals.z - row_max);
        exp_vals.w = expf(vals.w - row_max);
        
        reinterpret_cast<float4*>(row_data + i)[0] = exp_vals;
        local_sum += exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (lane == 0) {
        row_sum_shared[warp_id] = local_sum;
    }
    __syncthreads(); 
    
    if (warp_id == 0) {
        float val = (lane < num_warps) ? row_sum_shared[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            row_sum_shared[0] = val;
        }
    }
    __syncthreads();
    
    float row_sum = row_sum_shared[0];
    float inv_sum = 1.0f / row_sum; 
    
    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        float4 vals = reinterpret_cast<const float4*>(row_data + i)[0];
        vals.x *= inv_sum;
        vals.y *= inv_sum;
        vals.z *= inv_sum;
        vals.w *= inv_sum;
        reinterpret_cast<float4*>(row_data + i)[0] = vals;
    }
}

__global__ void softmax_kernel_multi_row_inplace(
    float* __restrict__ data,
    int row_count,
    int row_size
) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_idx >= row_count) return;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    
    float* row_data = data + row_idx * row_size;
    
    extern __shared__ float dynamic_shared_mem[];
    float* shared_max = dynamic_shared_mem;
    float* shared_sum = &dynamic_shared_mem[blockDim.y];
    
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_data[i]);
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (lane == 0) {
        shared_max[threadIdx.y] = local_max;
    }
    __syncthreads();
    
    float row_max = shared_max[threadIdx.y];
    
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float exp_val = expf(row_data[i] - row_max);
        row_data[i] = exp_val;
        local_sum += exp_val;
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (lane == 0) {
        shared_sum[threadIdx.y] = local_sum;
    }
    __syncthreads();
    
    float row_sum = shared_sum[threadIdx.y];
    float inv_sum = 1.0f / row_sum;
    
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

struct SoftmaxState {
    float* d_data = nullptr;
    size_t allocated_elements = 0;
    std::vector<float> result;
    
    ~SoftmaxState() {
        if (d_data) cudaFree(d_data);
    }
};

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    static SoftmaxState state;
    
    int total_elements = input.size();
    int row_size = total_elements / row_count;
    
    assert(total_elements % row_count == 0);
    assert(row_size > 0);
    
    if (state.allocated_elements != total_elements) {
        state.result.resize(total_elements);
        
        if (state.d_data) {
            cudaFree(state.d_data);
        }
        
        cudaMalloc(&state.d_data, total_elements * sizeof(float));
        state.allocated_elements = total_elements;
    }
    
    cudaMemcpy(state.d_data, input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    if (row_size <= 256) {
        int rows_per_block = std::min(8, row_count);
        dim3 block_dim(WARP_SIZE, rows_per_block, 1);
        dim3 grid_dim((row_count + rows_per_block - 1) / rows_per_block, 1, 1);
        
        int shared_mem_size = rows_per_block * 2 * sizeof(float);
        
        softmax_kernel_multi_row_inplace<<<grid_dim, block_dim, shared_mem_size>>>(
            state.d_data, row_count, row_size
        );
    } else {
        int threads_per_block;
        if (row_size >= 16384) {
            threads_per_block = 256;
        } else if (row_size >= 4096) {
            threads_per_block = 192;
        } else {
            threads_per_block = 128;
        }
        
        dim3 block_dim(threads_per_block, 1, 1);
        dim3 grid_dim(row_count, 1, 1);
        
        int shared_mem_size = (threads_per_block / WARP_SIZE) * 2 * sizeof(float);
        
        softmax_kernel_optimized_inplace<<<grid_dim, block_dim, shared_mem_size>>>(
            state.d_data, row_count, row_size
        );
    }
    
    cudaDeviceSynchronize();
    cudaMemcpy(state.result.data(), state.d_data, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    return state.result;
}