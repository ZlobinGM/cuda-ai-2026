import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

source = SourceModule("""
__global__ void getRowMeans(float* mat_cpu, int m, int n, float* rowMeans) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    float sum = 0.f;
    for (int j = 0; j < n; ++j) {
        sum += mat_cpu[i * n + j];
    }
    rowMeans[i] = sum / n;
}

__global__ void subCol(float* mat_cpu, int m, int n, float* col) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;

    mat_cpu[i * n + j] -= col[i];
}

__global__ void getRowVars(float* mat_cpu, int m, int n, float* rowVars) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    float sum = 0.f;
    float x;
    for (int j = 0; j < n; ++j) {
        x = mat_cpu[i * n + j];
        sum += x * x;
    }
    rowVars[i] = sum / n;
}

__global__ void invSqrt(float* x, int n, float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Используем sqrt вместо std::sqrt
    x[i] = 1.f / sqrt(x[i] + eps);
}

__global__ void applyVarGammaBeta(float* mat_cpu, int m, int n, float* invSqrtVar, float* gamma, float* beta) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;

    float x = invSqrtVar[i] * gamma[j];
    mat_cpu[i * n + j] = mat_cpu[i * n + j] * x + beta[j];
}
""")

get_row_means = source.get_function("getRowMeans")
sub_col = source.get_function("subCol")
get_row_vars = source.get_function("getRowVars")
inv_sqrt = source.get_function("invSqrt")
apply_var_gamma_beta = source.get_function("applyVarGammaBeta")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    mat_cpu = np.asarray(input, dtype=np.float32)
    gamma_cpu = np.asarray(gamma, dtype=np.float32)
    beta_cpu = np.asarray(beta, dtype=np.float32)

    m = np.int32(mat_cpu.size / row_size)
    n = np.int32(row_size)

    mat_gpu = cuda.mem_alloc(mat_cpu.nbytes)
    cuda.memcpy_htod(mat_gpu, mat_cpu)

    gamma_gpu = cuda.mem_alloc(gamma_cpu.nbytes)
    cuda.memcpy_htod(gamma_gpu, gamma_cpu)

    beta_gpu = cuda.mem_alloc(beta_cpu.nbytes)
    cuda.memcpy_htod(beta_gpu, beta_cpu)

    col_gpu = cuda.mem_alloc(int(m * 4))

    block_size_1d = (256, 1, 1)
    num_blocks_1d = (int((m + block_size_1d[0] - 1) // block_size_1d[0]), 1)

    block_size_2d = (16, 16, 1)
    num_blocks_2d = (
        int((n + block_size_2d[0] - 1) // block_size_2d[0]),
        int((m + block_size_2d[1] - 1) // block_size_2d[1]),
        1,
    )

    get_row_means(
        mat_gpu,
        m,
        n,
        col_gpu,
        block=block_size_1d,
        grid=num_blocks_1d,
    )

    sub_col(
        mat_gpu,
        m,
        n,
        col_gpu,
        block=block_size_2d,
        grid=num_blocks_2d,
    )

    get_row_vars(
        mat_gpu,
        m,
        n,
        col_gpu,
        block=block_size_1d,
        grid=num_blocks_1d,
    )

    inv_sqrt(
        col_gpu,
        m,
        np.float32(eps),
        block=block_size_1d,
        grid=num_blocks_1d,
    )

    apply_var_gamma_beta(
        mat_gpu,
        m,
        n,
        col_gpu,
        gamma_gpu,
        beta_gpu,
        block=block_size_2d,
        grid=num_blocks_2d,
    )

    out = np.empty_like(mat_cpu)
    cuda.memcpy_dtoh(out, mat_gpu)
    return out
