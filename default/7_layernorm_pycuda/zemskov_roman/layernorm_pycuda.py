import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

cuda_kernel_code = """
#define WARP_SIZE 32
#define VECTOR_SIZE 4

__global__ void layernorm_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_count,
    int row_size,
    float eps
) {
    extern __shared__ float shared_mem[];
    float* row_mean_shared = shared_mem;
    float* row_var_shared = &shared_mem[blockDim.x / WARP_SIZE];
    
    int row_idx = blockIdx.x;
    if (row_idx >= row_count) return;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    const float* input_row = input + row_idx * row_size;
    float* output_row = output + row_idx * row_size;
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        float4 vals = reinterpret_cast<const float4*>(input_row + i)[0];
        
        local_sum += vals.x + vals.y + vals.z + vals.w;
        local_sq_sum += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
    }
    
    if (lane == 0) {
        row_mean_shared[warp_id] = local_sum;
        row_var_shared[warp_id] = local_sq_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float sum_val = (lane < num_warps) ? row_mean_shared[lane] : 0.0f;
        float sq_sum_val = (lane < num_warps) ? row_var_shared[lane] : 0.0f;
        
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
            sq_sum_val += __shfl_down_sync(0xffffffff, sq_sum_val, offset);
        }
        
        if (lane == 0) {
            row_mean_shared[0] = sum_val;
            row_var_shared[0] = sq_sum_val;
        }
    }
    __syncthreads();
    
    float row_mean = row_mean_shared[0] / row_size;
    float row_var = (row_var_shared[0] / row_size) - (row_mean * row_mean);
    float inv_std = rsqrtf(row_var + eps);
    
    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        float4 vals = reinterpret_cast<const float4*>(input_row + i)[0];
        float4 gamma_vals = reinterpret_cast<const float4*>(gamma + i)[0];
        float4 beta_vals = reinterpret_cast<const float4*>(beta + i)[0];
        
        float4 out_vals;
        out_vals.x = ((vals.x - row_mean) * inv_std) * gamma_vals.x + beta_vals.x;
        out_vals.y = ((vals.y - row_mean) * inv_std) * gamma_vals.y + beta_vals.y;
        out_vals.z = ((vals.z - row_mean) * inv_std) * gamma_vals.z + beta_vals.z;
        out_vals.w = ((vals.w - row_mean) * inv_std) * gamma_vals.w + beta_vals.w;
        
        reinterpret_cast<float4*>(output_row + i)[0] = out_vals;
    }
}
"""

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    WARP_SIZE = 32
    if not hasattr(layernorm_pycuda, "mod"):
        layernorm_pycuda.mod = SourceModule(cuda_kernel_code,  options=["-O3", "-use_fast_math"])
        layernorm_pycuda.kernel = layernorm_pycuda.mod.get_function("layernorm_kernel_optimized")
        layernorm_pycuda.allocated_size = 0
        layernorm_pycuda.d_input = None
        layernorm_pycuda.d_output = None
        layernorm_pycuda.d_gamma = None
        layernorm_pycuda.d_beta = None

    input_arr = np.ascontiguousarray(input, dtype=np.float32)
    gamma_arr = np.ascontiguousarray(gamma, dtype=np.float32)
    beta_arr = np.ascontiguousarray(beta, dtype=np.float32)
    
    total_elements = input_arr.size
    row_count = total_elements // row_size
    
    if layernorm_pycuda.allocated_size != total_elements:
        if layernorm_pycuda.d_input is not None:
            layernorm_pycuda.d_input.free()
            layernorm_pycuda.d_output.free()
            layernorm_pycuda.d_gamma.free()
            layernorm_pycuda.d_beta.free()
            
        layernorm_pycuda.d_input = cuda.mem_alloc(input_arr.nbytes)
        layernorm_pycuda.d_output = cuda.mem_alloc(input_arr.nbytes)
        layernorm_pycuda.d_gamma = cuda.mem_alloc(gamma_arr.nbytes)
        layernorm_pycuda.d_beta = cuda.mem_alloc(beta_arr.nbytes)
        layernorm_pycuda.allocated_size = total_elements

    cuda.memcpy_htod(layernorm_pycuda.d_input, input_arr)
    cuda.memcpy_htod(layernorm_pycuda.d_gamma, gamma_arr)
    cuda.memcpy_htod(layernorm_pycuda.d_beta, beta_arr)
    
    if row_size >= 16384:
        threads_per_block = 256
    elif row_size >= 4096:
        threads_per_block = 192
    else:
        threads_per_block = 128
        
    shared_mem_size = (threads_per_block // WARP_SIZE) * 2 * 4
    
    layernorm_pycuda.kernel(
        layernorm_pycuda.d_input,
        layernorm_pycuda.d_gamma,
        layernorm_pycuda.d_beta,
        layernorm_pycuda.d_output,
        np.int32(row_count),
        np.int32(row_size),
        np.float32(eps),
        block=(threads_per_block, 1, 1),
        grid=(row_count, 1, 1),
        shared=shared_mem_size
    )
    
    output_arr = np.empty_like(input_arr)
    cuda.memcpy_dtoh(output_arr, layernorm_pycuda.d_output)
    
    return output_arr