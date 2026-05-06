// TurboQuant CUDA device functions and kernel declarations
// Port of ggml-turboquant.c to CUDA __device__ functions
// Algorithm: PolarQuant (b-1 bits) + QJL (1 bit) = b bits total per coordinate

#pragma once

#include "common.cuh"

// ============================================================================
// Constants
// ============================================================================

#define TURBO_QJL_CONST   1.2533141373f  // sqrt(pi/2)
#define TURBO_QJL_SEED_OFFSET 1000

// ============================================================================
// PRNG: splitmix64 — identical to C implementation
// ============================================================================

static __device__ __forceinline__ uint64_t turbo_splitmix64(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Sign generator: extracts one random sign per call from 64-bit PRNG words
struct turbo_sign_gen_t {
    uint64_t state;
    uint64_t bits;
    int remaining;
};

static __device__ __forceinline__ void turbo_sign_gen_init(turbo_sign_gen_t * sg, uint32_t seed) {
    sg->state = (uint64_t)seed ^ 0x5DEECE66DULL;
    sg->bits = turbo_splitmix64(&sg->state);
    sg->remaining = 64;
}

static __device__ __forceinline__ float turbo_sign_gen_next(turbo_sign_gen_t * sg) {
    if (sg->remaining == 0) {
        sg->bits = turbo_splitmix64(&sg->state);
        sg->remaining = 64;
    }
    float s = (sg->bits & 1) ? 1.0f : -1.0f;
    sg->bits >>= 1;
    sg->remaining--;
    return s;
}

// ============================================================================
// Codebook centroids (compile-time constants, same as C)
// ============================================================================

__constant__ static const float turbo_centroids_1bit_d[2] = {
    -0.1410473959f, 0.1410473959f
};

__constant__ static const float turbo_centroids_2bit_d[4] = {
    -0.2669328099f, -0.0800798430f, 0.0800798430f, 0.2669328099f
};

__constant__ static const float turbo_centroids_3bit_d[8] = {
    -0.3804137235f, -0.2375717128f, -0.1336440212f, -0.0433269046f,
     0.0433269046f,  0.1336440212f,  0.2375717128f,  0.3804137235f
};

__constant__ static const float turbo_boundaries_2bit_d[3] = {
    -0.1735063264f, 0.0f, 0.1735063264f
};

__constant__ static const float turbo_boundaries_3bit_d[7] = {
    -0.3089927181f, -0.1856078670f, -0.0884854629f, 0.0f,
     0.0884854629f,  0.1856078670f,  0.3089927181f
};

// ============================================================================
// Nearest centroid index lookup
// ============================================================================

static __device__ __forceinline__ int turbo_nearest_1bit(float val) {
    return val >= 0.0f ? 1 : 0;
}

static __device__ __forceinline__ int turbo_nearest_2bit(float val) {
    if (val < turbo_boundaries_2bit_d[1]) {
        return val < turbo_boundaries_2bit_d[0] ? 0 : 1;
    } else {
        return val < turbo_boundaries_2bit_d[2] ? 2 : 3;
    }
}

static __device__ __forceinline__ int turbo_nearest_3bit(float val) {
    if (val < turbo_boundaries_3bit_d[3]) {
        if (val < turbo_boundaries_3bit_d[1]) {
            return val < turbo_boundaries_3bit_d[0] ? 0 : 1;
        } else {
            return val < turbo_boundaries_3bit_d[2] ? 2 : 3;
        }
    } else {
        if (val < turbo_boundaries_3bit_d[5]) {
            return val < turbo_boundaries_3bit_d[4] ? 4 : 5;
        } else {
            return val < turbo_boundaries_3bit_d[6] ? 6 : 7;
        }
    }
}

// ============================================================================
// Walsh-Hadamard Transform: in-place butterfly for QK_TURBO=32
// ============================================================================

static __device__ __forceinline__ void turbo_wht32(float * x) {
    for (int len = 1; len < QK_TURBO; len <<= 1) {
        for (int i = 0; i < QK_TURBO; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    const float scale = 1.0f / sqrtf((float)QK_TURBO);
    for (int i = 0; i < QK_TURBO; i++) {
        x[i] *= scale;
    }
}

// ============================================================================
// Structured rotation: D2 @ WHT @ D1 and its inverse
// ============================================================================

static __device__ void turbo_rotation_forward(const float * src, float * dst, uint32_t seed) {
    turbo_sign_gen_t sg;
    turbo_sign_gen_init(&sg, seed);

    float d1[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        d1[i] = turbo_sign_gen_next(&sg);
        dst[i] = src[i] * d1[i];
    }

    turbo_wht32(dst);

    for (int i = 0; i < QK_TURBO; i++) {
        dst[i] *= turbo_sign_gen_next(&sg);
    }
}

static __device__ void turbo_rotation_inverse(const float * src, float * dst, uint32_t seed) {
    turbo_sign_gen_t sg;
    turbo_sign_gen_init(&sg, seed);

    float d1[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        d1[i] = turbo_sign_gen_next(&sg);
    }

    for (int i = 0; i < QK_TURBO; i++) {
        dst[i] = src[i] * turbo_sign_gen_next(&sg);
    }

    turbo_wht32(dst);

    for (int i = 0; i < QK_TURBO; i++) {
        dst[i] *= d1[i];
    }
}

// ============================================================================
// Seed generation (same as C)
// ============================================================================

static __device__ __forceinline__ uint32_t turbo_make_seed(int64_t block_idx) {
    uint64_t h = (uint64_t)block_idx * 0x9E3779B97F4A7C15ULL;
    return (uint32_t)(h ^ (h >> 32));
}

// ============================================================================
// Bit packing/unpacking
// ============================================================================

static __device__ void turbo_unpack_3bit(const uint8_t * packed, int * indices) {
    int bit_pos = 0;
    for (int i = 0; i < QK_TURBO; i++) {
        int val = 0;
        for (int b = 0; b < 3; b++) {
            if (packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                val |= (1 << b);
            }
            bit_pos++;
        }
        indices[i] = val;
    }
}

static __device__ void turbo_pack_3bit(const int * indices, uint8_t * packed) {
    for (int i = 0; i < QK_TURBO * 3 / 8; i++) packed[i] = 0;
    int bit_pos = 0;
    for (int i = 0; i < QK_TURBO; i++) {
        int val = indices[i] & 0x7;
        for (int b = 0; b < 3; b++) {
            if (val & (1 << b)) {
                packed[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            bit_pos++;
        }
    }
}

static __device__ void turbo_unpack_2bit(const uint8_t * packed, int * indices) {
    for (int i = 0; i < QK_TURBO; i++) {
        indices[i] = (packed[i / 4] >> ((i % 4) * 2)) & 0x3;
    }
}

static __device__ void turbo_pack_2bit(const int * indices, uint8_t * packed) {
    for (int i = 0; i < QK_TURBO * 2 / 8; i++) packed[i] = 0;
    for (int i = 0; i < QK_TURBO; i++) {
        packed[i / 4] |= (uint8_t)((indices[i] & 0x3) << ((i % 4) * 2));
    }
}

static __device__ void turbo_unpack_1bit(const uint8_t * packed, int * indices) {
    for (int i = 0; i < QK_TURBO; i++) {
        indices[i] = (packed[i / 8] >> (i % 8)) & 1;
    }
}

static __device__ void turbo_pack_1bit(const int * indices, uint8_t * packed) {
    for (int i = 0; i < QK_TURBO / 8; i++) packed[i] = 0;
    for (int i = 0; i < QK_TURBO; i++) {
        if (indices[i] & 1) {
            packed[i / 8] |= (1 << (i % 8));
        }
    }
}

// ============================================================================
// QJL quantize/dequantize
// ============================================================================

static __device__ void turbo_qjl_quantize(const float * residual, uint32_t seed, uint8_t * signs_out) {
    float projected[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) projected[i] = residual[i];

    turbo_wht32(projected);

    turbo_sign_gen_t sg;
    turbo_sign_gen_init(&sg, seed + TURBO_QJL_SEED_OFFSET);

    for (int i = 0; i < QK_TURBO / 8; i++) signs_out[i] = 0;
    for (int i = 0; i < QK_TURBO; i++) {
        float val = projected[i] * turbo_sign_gen_next(&sg);
        if (val >= 0.0f) {
            signs_out[i / 8] |= (1 << (i % 8));
        }
    }
}

static __device__ void turbo_qjl_dequantize(const uint8_t * signs_packed, float res_norm, uint32_t seed, float * out) {
    float signs_float[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        int bit = (signs_packed[i / 8] >> (i % 8)) & 1;
        signs_float[i] = bit ? 1.0f : -1.0f;
    }

    turbo_sign_gen_t sg;
    turbo_sign_gen_init(&sg, seed + TURBO_QJL_SEED_OFFSET);
    for (int i = 0; i < QK_TURBO; i++) {
        signs_float[i] *= turbo_sign_gen_next(&sg);
    }

    turbo_wht32(signs_float);

    float scale = TURBO_QJL_CONST / (float)QK_TURBO * res_norm;
    for (int i = 0; i < QK_TURBO; i++) {
        out[i] = signs_float[i] * scale;
    }
}

// ============================================================================
// Full block dequantize (device functions for each bit width)
// Output: QK_TURBO floats written to dst
// ============================================================================

static __device__ void turbo4_dequantize_block(const block_turbo4 * blk, float * dst) {
    float norm     = __half2float(blk->norm);
    float res_norm = __half2float(blk->res_norm);
    uint32_t seed  = blk->rot_seed;

    // Unpack 3-bit indices and look up centroids
    int indices[QK_TURBO];
    turbo_unpack_3bit(blk->qs, indices);

    float y_hat[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        y_hat[i] = turbo_centroids_3bit_d[indices[i]];
    }

    // Norm correction
    float y_hat_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) y_hat_norm_sq += y_hat[i] * y_hat[i];
    float y_hat_norm = sqrtf(y_hat_norm_sq);
    if (y_hat_norm > 1e-10f) {
        float inv = 1.0f / y_hat_norm;
        for (int i = 0; i < QK_TURBO; i++) y_hat[i] *= inv;
    }

    // Inverse rotation and rescale
    float x_pq[QK_TURBO];
    turbo_rotation_inverse(y_hat, x_pq, seed);
    for (int i = 0; i < QK_TURBO; i++) x_pq[i] *= norm;

    // QJL residual reconstruction
    float residual[QK_TURBO];
    turbo_qjl_dequantize(blk->signs, res_norm, seed, residual);

    // Sum
    for (int i = 0; i < QK_TURBO; i++) dst[i] = x_pq[i] + residual[i];
}

static __device__ void turbo3_dequantize_block(const block_turbo3 * blk, float * dst) {
    float norm     = __half2float(blk->norm);
    float res_norm = __half2float(blk->res_norm);
    uint32_t seed  = blk->rot_seed;

    int indices[QK_TURBO];
    turbo_unpack_2bit(blk->qs, indices);

    float y_hat[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) y_hat[i] = turbo_centroids_2bit_d[indices[i]];

    float y_hat_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) y_hat_norm_sq += y_hat[i] * y_hat[i];
    float y_hat_norm = sqrtf(y_hat_norm_sq);
    if (y_hat_norm > 1e-10f) {
        float inv = 1.0f / y_hat_norm;
        for (int i = 0; i < QK_TURBO; i++) y_hat[i] *= inv;
    }

    float x_pq[QK_TURBO];
    turbo_rotation_inverse(y_hat, x_pq, seed);
    for (int i = 0; i < QK_TURBO; i++) x_pq[i] *= norm;

    float residual[QK_TURBO];
    turbo_qjl_dequantize(blk->signs, res_norm, seed, residual);

    for (int i = 0; i < QK_TURBO; i++) dst[i] = x_pq[i] + residual[i];
}

static __device__ void turbo2_dequantize_block(const block_turbo2 * blk, float * dst) {
    float norm     = __half2float(blk->norm);
    float res_norm = __half2float(blk->res_norm);
    uint32_t seed  = blk->rot_seed;

    int indices[QK_TURBO];
    turbo_unpack_1bit(blk->qs, indices);

    float y_hat[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) y_hat[i] = turbo_centroids_1bit_d[indices[i]];

    float y_hat_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) y_hat_norm_sq += y_hat[i] * y_hat[i];
    float y_hat_norm = sqrtf(y_hat_norm_sq);
    if (y_hat_norm > 1e-10f) {
        float inv = 1.0f / y_hat_norm;
        for (int i = 0; i < QK_TURBO; i++) y_hat[i] *= inv;
    }

    float x_pq[QK_TURBO];
    turbo_rotation_inverse(y_hat, x_pq, seed);
    for (int i = 0; i < QK_TURBO; i++) x_pq[i] *= norm;

    float residual[QK_TURBO];
    turbo_qjl_dequantize(blk->signs, res_norm, seed, residual);

    for (int i = 0; i < QK_TURBO; i++) dst[i] = x_pq[i] + residual[i];
}

// ============================================================================
// Full block quantize (device functions for set_rows)
// Input: QK_TURBO floats, Output: one packed block
// ============================================================================

static __device__ void turbo4_quantize_block(const float * src, block_turbo4 * blk, int64_t block_idx) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-30f) ? 1.0f / norm : 0.0f;

    float x_norm[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) x_norm[i] = src[i] * inv_norm;

    uint32_t seed = turbo_make_seed(block_idx);
    float rotated[QK_TURBO];
    turbo_rotation_forward(x_norm, rotated, seed);

    int indices[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) indices[i] = turbo_nearest_3bit(rotated[i]);

    float y_hat[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) y_hat[i] = turbo_centroids_3bit_d[indices[i]];

    float y_hat_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) y_hat_norm_sq += y_hat[i] * y_hat[i];
    float y_hat_norm = sqrtf(y_hat_norm_sq);
    if (y_hat_norm > 1e-10f) {
        float inv = 1.0f / y_hat_norm;
        for (int i = 0; i < QK_TURBO; i++) y_hat[i] *= inv;
    }

    float x_pq[QK_TURBO];
    turbo_rotation_inverse(y_hat, x_pq, seed);
    for (int i = 0; i < QK_TURBO; i++) x_pq[i] *= norm;

    float residual[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) residual[i] = src[i] - x_pq[i];

    float res_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) res_norm_sq += residual[i] * residual[i];
    float res_norm = sqrtf(res_norm_sq);

    uint8_t qjl_signs[QK_TURBO / 8];
    turbo_qjl_quantize(residual, seed, qjl_signs);

    blk->norm     = __float2half(norm);
    blk->res_norm = __float2half(res_norm);
    blk->rot_seed = seed;
    turbo_pack_3bit(indices, blk->qs);
    for (int i = 0; i < QK_TURBO / 8; i++) blk->signs[i] = qjl_signs[i];
}

static __device__ void turbo3_quantize_block(const float * src, block_turbo3 * blk, int64_t block_idx) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-30f) ? 1.0f / norm : 0.0f;

    float x_norm[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) x_norm[i] = src[i] * inv_norm;

    uint32_t seed = turbo_make_seed(block_idx);
    float rotated[QK_TURBO];
    turbo_rotation_forward(x_norm, rotated, seed);

    int indices[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) indices[i] = turbo_nearest_2bit(rotated[i]);

    float y_hat[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) y_hat[i] = turbo_centroids_2bit_d[indices[i]];

    float y_hat_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) y_hat_norm_sq += y_hat[i] * y_hat[i];
    float y_hat_norm = sqrtf(y_hat_norm_sq);
    if (y_hat_norm > 1e-10f) {
        float inv = 1.0f / y_hat_norm;
        for (int i = 0; i < QK_TURBO; i++) y_hat[i] *= inv;
    }

    float x_pq[QK_TURBO];
    turbo_rotation_inverse(y_hat, x_pq, seed);
    for (int i = 0; i < QK_TURBO; i++) x_pq[i] *= norm;

    float residual[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) residual[i] = src[i] - x_pq[i];

    float res_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) res_norm_sq += residual[i] * residual[i];
    float res_norm = sqrtf(res_norm_sq);

    uint8_t qjl_signs[QK_TURBO / 8];
    turbo_qjl_quantize(residual, seed, qjl_signs);

    blk->norm     = __float2half(norm);
    blk->res_norm = __float2half(res_norm);
    blk->rot_seed = seed;
    turbo_pack_2bit(indices, blk->qs);
    for (int i = 0; i < QK_TURBO / 8; i++) blk->signs[i] = qjl_signs[i];
}

static __device__ void turbo2_quantize_block(const float * src, block_turbo2 * blk, int64_t block_idx) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-30f) ? 1.0f / norm : 0.0f;

    float x_norm[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) x_norm[i] = src[i] * inv_norm;

    uint32_t seed = turbo_make_seed(block_idx);
    float rotated[QK_TURBO];
    turbo_rotation_forward(x_norm, rotated, seed);

    int indices[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) indices[i] = turbo_nearest_1bit(rotated[i]);

    float y_hat[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) y_hat[i] = turbo_centroids_1bit_d[indices[i]];

    float y_hat_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) y_hat_norm_sq += y_hat[i] * y_hat[i];
    float y_hat_norm = sqrtf(y_hat_norm_sq);
    if (y_hat_norm > 1e-10f) {
        float inv = 1.0f / y_hat_norm;
        for (int i = 0; i < QK_TURBO; i++) y_hat[i] *= inv;
    }

    float x_pq[QK_TURBO];
    turbo_rotation_inverse(y_hat, x_pq, seed);
    for (int i = 0; i < QK_TURBO; i++) x_pq[i] *= norm;

    float residual[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) residual[i] = src[i] - x_pq[i];

    float res_norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) res_norm_sq += residual[i] * residual[i];
    float res_norm = sqrtf(res_norm_sq);

    uint8_t qjl_signs[QK_TURBO / 8];
    turbo_qjl_quantize(residual, seed, qjl_signs);

    blk->norm     = __float2half(norm);
    blk->res_norm = __float2half(res_norm);
    blk->rot_seed = seed;
    turbo_pack_1bit(indices, blk->qs);
    for (int i = 0; i < QK_TURBO / 8; i++) blk->signs[i] = qjl_signs[i];
}

// ============================================================================
// Quantize block wrappers for set_rows template (no block_idx param)
// Uses a flat pointer-derived block index for seed generation.
// ============================================================================

static __device__ void quantize_f32_turbo4_block(const float * x, block_turbo4 * y) {
    // Derive block index from dst pointer — each thread processes one block
    // The seed is deterministic per block position, not per pointer
    int64_t block_idx = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x);
    turbo4_quantize_block(x, y, block_idx);
}

static __device__ void quantize_f32_turbo3_block(const float * x, block_turbo3 * y) {
    int64_t block_idx = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x);
    turbo3_quantize_block(x, y, block_idx);
}

static __device__ void quantize_f32_turbo2_block(const float * x, block_turbo2 * y) {
    int64_t block_idx = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x);
    turbo2_quantize_block(x, y, block_idx);
}

// ============================================================================
// Kernel declarations (implemented in turbo-quant.cu)
// ============================================================================

// Dequantize full rows: turbo blocks → float
void dequantize_row_turbo4_cuda(const void * x, float * y, int64_t k, cudaStream_t stream);
void dequantize_row_turbo3_cuda(const void * x, float * y, int64_t k, cudaStream_t stream);
void dequantize_row_turbo2_cuda(const void * x, float * y, int64_t k, cudaStream_t stream);

// Custom mul_mat_vec: turbo (quantized src0 rows) × float32 (src1 vec) → float32 (dst)
void mul_mat_vec_turbo4_cuda(
    const void * src0, const float * src1, float * dst,
    int64_t ncols, int64_t nrows, cudaStream_t stream);
void mul_mat_vec_turbo3_cuda(
    const void * src0, const float * src1, float * dst,
    int64_t ncols, int64_t nrows, cudaStream_t stream);
void mul_mat_vec_turbo2_cuda(
    const void * src0, const float * src1, float * dst,
    int64_t ncols, int64_t nrows, cudaStream_t stream);

// Custom get_rows: turbo blocks → float (full-block dequant per row)
void get_rows_turbo4_cuda(
    const void * src0, const int32_t * src1, float * dst,
    int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);
void get_rows_turbo3_cuda(
    const void * src0, const int32_t * src1, float * dst,
    int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);
void get_rows_turbo2_cuda(
    const void * src0, const int32_t * src1, float * dst,
    int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);
