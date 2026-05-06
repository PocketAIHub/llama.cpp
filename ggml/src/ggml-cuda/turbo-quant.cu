// TurboQuant CUDA kernels: dequantize rows, mul_mat_vec
// These are custom kernels because TurboQuant requires full-block processing
// (Walsh-Hadamard Transform) which is incompatible with the standard
// per-element dequantize pattern used by q4_0/q8_0/etc.

#include "turbo-quant.cuh"

#define TURBO_DEQUANT_BLOCK_SIZE 256
#define TURBO_MUL_MV_BLOCK_SIZE  32  // One warp per row

// ============================================================================
// Dequantize row kernels: one thread per block of QK_TURBO values
// ============================================================================

template <typename block_type, void (*dequant_func)(const block_type *, float *)>
static __global__ void k_dequantize_row_turbo(
        const void * __restrict__ src, float * __restrict__ dst, int64_t nb) {
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= nb) return;

    const block_type * x = (const block_type *)src;
    dequant_func(&x[ib], dst + ib * QK_TURBO);
}

void dequantize_row_turbo4_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    const int64_t nb = k / QK_TURBO;
    const int grid = (nb + TURBO_DEQUANT_BLOCK_SIZE - 1) / TURBO_DEQUANT_BLOCK_SIZE;
    k_dequantize_row_turbo<block_turbo4, turbo4_dequantize_block>
        <<<grid, TURBO_DEQUANT_BLOCK_SIZE, 0, stream>>>(x, y, nb);
}

void dequantize_row_turbo3_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    const int64_t nb = k / QK_TURBO;
    const int grid = (nb + TURBO_DEQUANT_BLOCK_SIZE - 1) / TURBO_DEQUANT_BLOCK_SIZE;
    k_dequantize_row_turbo<block_turbo3, turbo3_dequantize_block>
        <<<grid, TURBO_DEQUANT_BLOCK_SIZE, 0, stream>>>(x, y, nb);
}

void dequantize_row_turbo2_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    const int64_t nb = k / QK_TURBO;
    const int grid = (nb + TURBO_DEQUANT_BLOCK_SIZE - 1) / TURBO_DEQUANT_BLOCK_SIZE;
    k_dequantize_row_turbo<block_turbo2, turbo2_dequantize_block>
        <<<grid, TURBO_DEQUANT_BLOCK_SIZE, 0, stream>>>(x, y, nb);
}

// ============================================================================
// Custom mul_mat_vec kernel: turbo × f32 → f32
//
// One thread block per output row. Each thread handles one or more blocks,
// dequantizes to float, computes partial dot product, then warp reduces.
//
// src0: quantized matrix (nrows × ncols), row-major blocks
// src1: float32 vector (ncols)
// dst:  float32 result (nrows)
// ============================================================================

template <typename block_type, void (*dequant_func)(const block_type *, float *)>
static __global__ void k_mul_mat_vec_turbo(
        const void * __restrict__ src0, const float * __restrict__ src1,
        float * __restrict__ dst,
        int64_t ncols, int64_t nrows) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) return;

    const int64_t nb_per_row = ncols / QK_TURBO;
    const block_type * x_row = (const block_type *)src0 + row * nb_per_row;

    float thread_sum = 0.0f;

    // Each thread in the warp handles blocks strided by warpSize
    for (int64_t ib = threadIdx.x; ib < nb_per_row; ib += blockDim.x) {
        float tmp[QK_TURBO];
        dequant_func(&x_row[ib], tmp);

        const float * s1 = src1 + ib * QK_TURBO;
        float dot = 0.0f;
        for (int j = 0; j < QK_TURBO; j++) {
            dot += tmp[j] * s1[j];
        }
        thread_sum += dot;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Thread 0 of each block writes the result
    if (threadIdx.x == 0) {
        dst[row] = thread_sum;
    }
}

void mul_mat_vec_turbo4_cuda(
        const void * src0, const float * src1, float * dst,
        int64_t ncols, int64_t nrows, cudaStream_t stream) {
    k_mul_mat_vec_turbo<block_turbo4, turbo4_dequantize_block>
        <<<nrows, TURBO_MUL_MV_BLOCK_SIZE, 0, stream>>>(src0, src1, dst, ncols, nrows);
}

void mul_mat_vec_turbo3_cuda(
        const void * src0, const float * src1, float * dst,
        int64_t ncols, int64_t nrows, cudaStream_t stream) {
    k_mul_mat_vec_turbo<block_turbo3, turbo3_dequantize_block>
        <<<nrows, TURBO_MUL_MV_BLOCK_SIZE, 0, stream>>>(src0, src1, dst, ncols, nrows);
}

void mul_mat_vec_turbo2_cuda(
        const void * src0, const float * src1, float * dst,
        int64_t ncols, int64_t nrows, cudaStream_t stream) {
    k_mul_mat_vec_turbo<block_turbo2, turbo2_dequantize_block>
        <<<nrows, TURBO_MUL_MV_BLOCK_SIZE, 0, stream>>>(src0, src1, dst, ncols, nrows);
}

// ============================================================================
// Custom get_rows kernel: one thread per block
// ============================================================================

template <typename block_type, void (*dequant_func)(const block_type *, float *)>
static __global__ void k_get_rows_turbo(
        const void * __restrict__ src0, const int32_t * __restrict__ src1,
        float * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    const int64_t nb_per_row = ne00 / QK_TURBO;

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t ib = blockIdx.y * blockDim.x + threadIdx.x;
        if (ib >= nb_per_row) return;

        const int i10 = blockIdx.x;
        const int i11 = z / ne12;
        const int i12 = z % ne12;

        const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

        float * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;
        const char * src0_row = (const char *)src0 + i01 * nb01 + i11 * nb02 + i12 * nb03;
        const block_type * x_row = (const block_type *)src0_row;

        float tmp[QK_TURBO];
        dequant_func(&x_row[ib], tmp);

        for (int j = 0; j < QK_TURBO; j++) {
            dst_row[ib * QK_TURBO + j] = tmp[j];
        }
    }
}

// Host dispatch functions for get_rows
void get_rows_turbo4_cuda(
        const void * src0, const int32_t * src1, float * dst,
        int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    const int64_t nb_per_row = ne00 / QK_TURBO;
    const int block_size = 256;
    const dim3 grid(ne10, (nb_per_row + block_size - 1) / block_size, ne11 * ne12);

    const size_t s1  = nb1  / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_turbo<block_turbo4, turbo4_dequantize_block>
        <<<grid, block_size, 0, stream>>>(
            src0, src1, dst, ne00, ne11, ne12,
            s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

void get_rows_turbo3_cuda(
        const void * src0, const int32_t * src1, float * dst,
        int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    const int64_t nb_per_row = ne00 / QK_TURBO;
    const int block_size = 256;
    const dim3 grid(ne10, (nb_per_row + block_size - 1) / block_size, ne11 * ne12);

    const size_t s1  = nb1  / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_turbo<block_turbo3, turbo3_dequantize_block>
        <<<grid, block_size, 0, stream>>>(
            src0, src1, dst, ne00, ne11, ne12,
            s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

void get_rows_turbo2_cuda(
        const void * src0, const int32_t * src1, float * dst,
        int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    const int64_t nb_per_row = ne00 / QK_TURBO;
    const int block_size = 256;
    const dim3 grid(ne10, (nb_per_row + block_size - 1) / block_size, ne11 * ne12);

    const size_t s1  = nb1  / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_turbo<block_turbo2, turbo2_dequantize_block>
        <<<grid, block_size, 0, stream>>>(
            src0, src1, dst, ne00, ne11, ne12,
            s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}
