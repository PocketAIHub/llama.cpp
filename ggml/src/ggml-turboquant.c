// TurboQuant KV cache compression: PolarQuant + QJL
// Reference: TurboQuant (ICLR 2026, arXiv:2504.19874)
// Python reference: TheTom/turboquant_plus
//
// Algorithm overview (per block of QK_TURBO=32 floats):
//
// QUANTIZE (turbo4 = 3-bit PolarQuant + 1-bit QJL):
//   1. Compute norm = ||x||_2, normalize: x_hat = x / norm
//   2. Generate deterministic sign vectors from rot_seed
//   3. Apply fast rotation: y = D2 @ WHT @ D1 @ x_hat
//   4. Quantize each y[i] to nearest 3-bit centroid → indices
//   5. Dequantize to get y_hat, inverse-rotate to get x_pq
//   6. Compute residual r = x - x_pq * norm
//   7. QJL: generate projection signs from seed, project r, take sign bits
//   8. Store: norm, residual_norm, rot_seed, packed indices, packed QJL signs
//
// DEQUANTIZE:
//   1. Unpack indices → look up centroids → y_hat
//   2. Norm-correct y_hat (renormalize to unit length)
//   3. Inverse-rotate: x_hat = D1 @ WHT @ D2 @ y_hat
//   4. Rescale: x_pq = x_hat * norm
//   5. Unpack QJL signs, generate same projection, reconstruct residual
//   6. x_reconstructed = x_pq + residual_reconstructed

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-turboquant.h"
#include "ggml-impl.h"

#include <string.h>
#include <assert.h>
#include <math.h>

// ============================================================================
// PRNG: splitmix64 — fast, deterministic, seeded from rot_seed
// Used to generate random sign vectors for WHT rotation and QJL projection.
// ============================================================================

static inline uint64_t splitmix64_next(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Generate a random sign (+1.0 or -1.0) from one bit of PRNG output.
// Consumes bits from a 64-bit word, refilling when exhausted.
typedef struct {
    uint64_t state;
    uint64_t bits;
    int remaining;
} sign_gen_t;

static inline void sign_gen_init(sign_gen_t * sg, uint32_t seed) {
    sg->state = (uint64_t)seed ^ 0x5DEECE66DULL;
    sg->bits = splitmix64_next(&sg->state);
    sg->remaining = 64;
}

static inline float sign_gen_next(sign_gen_t * sg) {
    if (sg->remaining == 0) {
        sg->bits = splitmix64_next(&sg->state);
        sg->remaining = 64;
    }
    float s = (sg->bits & 1) ? 1.0f : -1.0f;
    sg->bits >>= 1;
    sg->remaining--;
    return s;
}

// ============================================================================
// Codebook: precomputed optimal centroids for N(0, 1/d) where d = QK_TURBO = 32
//
// Generated from Python reference: turboquant.codebook.optimal_centroids()
// 1-bit: closed-form ±sqrt(2/(pi*d))
// 2-bit: paper Table 1, scaled by 1/sqrt(d)
// 3-bit: Lloyd's algorithm on N(0, 1/32), 100 iterations
//
// All centroid arrays are sorted ascending and symmetric around 0.
// ============================================================================

// 1-bit: 2 centroids (for turbo2: b=2, PolarQuant uses b-1=1 bit)
static const float turbo_centroids_1bit[2] = {
    -0.1410473959f, 0.1410473959f
};

// 2-bit: 4 centroids (for turbo3: b=3, PolarQuant uses b-1=2 bits)
static const float turbo_centroids_2bit[4] = {
    -0.2669328099f, -0.0800798430f, 0.0800798430f, 0.2669328099f
};

// 3-bit: 8 centroids (for turbo4: b=4, PolarQuant uses b-1=3 bits)
static const float turbo_centroids_3bit[8] = {
    -0.3804137235f, -0.2375717128f, -0.1336440212f, -0.0433269046f,
     0.0433269046f,  0.1336440212f,  0.2375717128f,  0.3804137235f
};

// Decision boundaries (midpoints between adjacent centroids) for fast nearest lookup
// 1-bit boundary is trivially 0.0 — handled inline in nearest_index_1bit()

static const float turbo_boundaries_2bit[3] = {
    -0.1735063264f, 0.0f, 0.1735063264f
};

static const float turbo_boundaries_3bit[7] = {
    -0.3089927181f, -0.1856078670f, -0.0884854629f, 0.0f,
     0.0884854629f,  0.1856078670f,  0.3089927181f
};

// ============================================================================
// Nearest centroid index lookup via sorted boundaries (binary search equivalent)
// For small codebooks (2-8 entries) a linear scan on boundaries is optimal.
// ============================================================================

static inline int nearest_index_1bit(float val) {
    return val >= 0.0f ? 1 : 0;
}

static inline int nearest_index_2bit(float val) {
    if (val < turbo_boundaries_2bit[1]) {
        return val < turbo_boundaries_2bit[0] ? 0 : 1;
    } else {
        return val < turbo_boundaries_2bit[2] ? 2 : 3;
    }
}

static inline int nearest_index_3bit(float val) {
    // Binary search on 7 boundaries for 8 bins
    if (val < turbo_boundaries_3bit[3]) {
        if (val < turbo_boundaries_3bit[1]) {
            return val < turbo_boundaries_3bit[0] ? 0 : 1;
        } else {
            return val < turbo_boundaries_3bit[2] ? 2 : 3;
        }
    } else {
        if (val < turbo_boundaries_3bit[5]) {
            return val < turbo_boundaries_3bit[4] ? 4 : 5;
        } else {
            return val < turbo_boundaries_3bit[6] ? 6 : 7;
        }
    }
}

// ============================================================================
// Walsh-Hadamard Transform: O(d log d) in-place butterfly
// Operates on QK_TURBO=32 elements. Normalized by 1/sqrt(32).
// ============================================================================

static void fast_walsh_hadamard_32(float * x) {
    // 5 stages for n=32 (log2(32) = 5)
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
    // Normalize: 1/sqrt(32)
    const float scale = 1.0f / sqrtf((float)QK_TURBO);
    for (int i = 0; i < QK_TURBO; i++) {
        x[i] *= scale;
    }
}

// ============================================================================
// Structured random rotation: D2 @ WHT @ D1
// D1, D2 are diagonal sign matrices generated from rot_seed.
// Forward rotation and its inverse (transpose) differ only in sign application order.
// ============================================================================

// Apply forward rotation: D2 @ WHT @ D1 @ x
// Writes result to dst (may alias src if desired, but we use separate buffers).
static void apply_rotation_forward(const float * src, float * dst, uint32_t seed) {
    sign_gen_t sg;
    sign_gen_init(&sg, seed);

    // Generate D1 signs and apply: dst = D1 @ src
    float d1_signs[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        d1_signs[i] = sign_gen_next(&sg);
        dst[i] = src[i] * d1_signs[i];
    }

    // WHT in-place
    fast_walsh_hadamard_32(dst);

    // Generate D2 signs and apply: dst = D2 @ WHT @ D1 @ src
    for (int i = 0; i < QK_TURBO; i++) {
        dst[i] *= sign_gen_next(&sg);
    }
}

// Apply inverse rotation: D1 @ WHT @ D2 @ y
// (WHT is self-transpose, D is self-transpose, so inverse = D1 @ WHT @ D2)
static void apply_rotation_inverse(const float * src, float * dst, uint32_t seed) {
    sign_gen_t sg;
    sign_gen_init(&sg, seed);

    // Regenerate D1 signs (need them for step 3, but generate in order)
    float d1_signs[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        d1_signs[i] = sign_gen_next(&sg);
    }

    // Generate D2 signs and apply first (reverse order): dst = D2 @ src
    for (int i = 0; i < QK_TURBO; i++) {
        dst[i] = src[i] * sign_gen_next(&sg);
    }

    // WHT in-place
    fast_walsh_hadamard_32(dst);

    // Apply D1: dst = D1 @ WHT @ D2 @ src
    for (int i = 0; i < QK_TURBO; i++) {
        dst[i] *= d1_signs[i];
    }
}

// ============================================================================
// QJL: Structured random projection using D @ WHT
// Instead of materializing a full d×d Gaussian matrix, we use the structured
// approach: project via D_qjl @ WHT, where D_qjl is a random sign diagonal.
// This gives the same statistical guarantees with O(d log d) compute.
//
// Quantize:  signs = sign(D_qjl @ WHT @ residual)
// Dequantize: r_hat = sqrt(pi/2) / d * res_norm * (D_qjl @ WHT)^T @ signs
//           = sqrt(pi/2) / d * res_norm * WHT @ D_qjl @ signs
// ============================================================================

#define QJL_CONST 1.2533141373f  // sqrt(pi/2)
#define QJL_SEED_OFFSET 1000     // QJL uses rot_seed + 1000 for independent projection

// QJL quantize: compute sign bits of structured projection of residual
static void qjl_quantize(const float * residual, uint32_t seed, uint8_t * signs_out) {
    float projected[QK_TURBO];

    // Copy residual
    memcpy(projected, residual, QK_TURBO * sizeof(float));

    // Apply WHT
    fast_walsh_hadamard_32(projected);

    // Apply D_qjl signs and take sign of result
    sign_gen_t sg;
    sign_gen_init(&sg, seed + QJL_SEED_OFFSET);

    // Pack sign bits: +1 → bit=1, -1 → bit=0
    memset(signs_out, 0, QK_TURBO / 8);
    for (int i = 0; i < QK_TURBO; i++) {
        float val = projected[i] * sign_gen_next(&sg);
        if (val >= 0.0f) {
            signs_out[i / 8] |= (1 << (i % 8));
        }
    }
}

// QJL dequantize: reconstruct approximate residual from sign bits
static void qjl_dequantize(const uint8_t * signs_packed, float res_norm, uint32_t seed, float * residual_out) {
    // Unpack sign bits to float: bit=1 → +1.0, bit=0 → -1.0
    float signs_float[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        int bit = (signs_packed[i / 8] >> (i % 8)) & 1;
        signs_float[i] = bit ? 1.0f : -1.0f;
    }

    // Apply D_qjl signs (same as during quantize)
    sign_gen_t sg;
    sign_gen_init(&sg, seed + QJL_SEED_OFFSET);
    for (int i = 0; i < QK_TURBO; i++) {
        signs_float[i] *= sign_gen_next(&sg);
    }

    // Apply WHT (inverse of structured projection)
    fast_walsh_hadamard_32(signs_float);

    // Scale: sqrt(pi/2) / d * res_norm
    float scale = QJL_CONST / (float)QK_TURBO * res_norm;
    for (int i = 0; i < QK_TURBO; i++) {
        residual_out[i] = signs_float[i] * scale;
    }
}

// ============================================================================
// Bit packing utilities for PolarQuant indices
// ============================================================================

// Pack 32 3-bit indices into 12 bytes (96 bits)
// Layout: sequential bit packing, LSB-first within each byte
static void pack_3bit_indices(const int * indices, uint8_t * packed) {
    memset(packed, 0, QK_TURBO * 3 / 8);  // 12 bytes
    int bit_pos = 0;
    for (int i = 0; i < QK_TURBO; i++) {
        int val = indices[i] & 0x7;  // 3 bits
        for (int b = 0; b < 3; b++) {
            if (val & (1 << b)) {
                packed[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            bit_pos++;
        }
    }
}

// Unpack 12 bytes into 32 3-bit indices
static void unpack_3bit_indices(const uint8_t * packed, int * indices) {
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

// Pack 32 2-bit indices into 8 bytes
static void pack_2bit_indices(const int * indices, uint8_t * packed) {
    memset(packed, 0, QK_TURBO * 2 / 8);  // 8 bytes
    for (int i = 0; i < QK_TURBO; i++) {
        int val = indices[i] & 0x3;
        int byte_idx = i / 4;
        int bit_offset = (i % 4) * 2;
        packed[byte_idx] |= (uint8_t)(val << bit_offset);
    }
}

// Unpack 8 bytes into 32 2-bit indices
static void unpack_2bit_indices(const uint8_t * packed, int * indices) {
    for (int i = 0; i < QK_TURBO; i++) {
        int byte_idx = i / 4;
        int bit_offset = (i % 4) * 2;
        indices[i] = (packed[byte_idx] >> bit_offset) & 0x3;
    }
}

// Pack 32 1-bit indices into 4 bytes
static void pack_1bit_indices(const int * indices, uint8_t * packed) {
    memset(packed, 0, QK_TURBO / 8);  // 4 bytes
    for (int i = 0; i < QK_TURBO; i++) {
        if (indices[i] & 1) {
            packed[i / 8] |= (1 << (i % 8));
        }
    }
}

// Unpack 4 bytes into 32 1-bit indices
static void unpack_1bit_indices(const uint8_t * packed, int * indices) {
    for (int i = 0; i < QK_TURBO; i++) {
        indices[i] = (packed[i / 8] >> (i % 8)) & 1;
    }
}

// ============================================================================
// Seed generation: monotonically increasing per block for determinism
// Using block index directly — each block in a row gets a unique seed.
// ============================================================================

static inline uint32_t make_seed(int64_t block_idx) {
    // Mix the block index to avoid correlated seeds between adjacent blocks.
    // Simple but effective: multiply by golden ratio constant.
    uint64_t h = (uint64_t)block_idx * 0x9E3779B97F4A7C15ULL;
    return (uint32_t)(h ^ (h >> 32));
}

// ============================================================================
// turbo4: 3-bit PolarQuant + 1-bit QJL = 4 bits/val
// ============================================================================

void quantize_row_turbo4_ref(const float * GGML_RESTRICT x, block_turbo4 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k / QK_TURBO;

    for (int64_t ib = 0; ib < nb; ib++) {
        const float * xb = x + ib * QK_TURBO;
        block_turbo4 * blk = &y[ib];

        // --- Step 1: Compute L2 norm and normalize ---
        float norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            norm_sq += xb[i] * xb[i];
        }
        float norm = sqrtf(norm_sq);
        float inv_norm = (norm > 1e-30f) ? 1.0f / norm : 0.0f;

        float x_normalized[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            x_normalized[i] = xb[i] * inv_norm;
        }

        // --- Step 2: Generate seed and apply forward rotation ---
        uint32_t seed = make_seed(ib);
        float rotated[QK_TURBO];
        apply_rotation_forward(x_normalized, rotated, seed);

        // --- Step 3: Quantize to nearest 3-bit centroid ---
        int indices[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            indices[i] = nearest_index_3bit(rotated[i]);
        }

        // --- Step 4: Dequantize in rotated domain for residual computation ---
        float y_hat[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat[i] = turbo_centroids_3bit[indices[i]];
        }

        // Norm correction: renormalize y_hat to unit length before inverse rotation
        float y_hat_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat_norm_sq += y_hat[i] * y_hat[i];
        }
        float y_hat_norm = sqrtf(y_hat_norm_sq);
        if (y_hat_norm > 1e-10f) {
            float inv_y_hat_norm = 1.0f / y_hat_norm;
            for (int i = 0; i < QK_TURBO; i++) {
                y_hat[i] *= inv_y_hat_norm;
            }
        }

        // --- Step 5: Inverse rotate to get PolarQuant reconstruction ---
        float x_pq[QK_TURBO];
        apply_rotation_inverse(y_hat, x_pq, seed);

        // Rescale by original norm
        for (int i = 0; i < QK_TURBO; i++) {
            x_pq[i] *= norm;
        }

        // --- Step 6: Compute residual ---
        float residual[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            residual[i] = xb[i] - x_pq[i];
        }

        float res_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            res_norm_sq += residual[i] * residual[i];
        }
        float res_norm = sqrtf(res_norm_sq);

        // --- Step 7: QJL on residual ---
        uint8_t qjl_signs[QK_TURBO / 8];
        qjl_quantize(residual, seed, qjl_signs);

        // --- Step 8: Pack into block ---
        blk->norm     = GGML_FP32_TO_FP16(norm);
        blk->res_norm = GGML_FP32_TO_FP16(res_norm);
        blk->rot_seed = seed;
        pack_3bit_indices(indices, blk->qs);
        memcpy(blk->signs, qjl_signs, QK_TURBO / 8);
    }
}

void dequantize_row_turbo4(const block_turbo4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k / QK_TURBO;

    for (int64_t ib = 0; ib < nb; ib++) {
        const block_turbo4 * blk = &x[ib];
        float * yb = y + ib * QK_TURBO;

        float norm     = GGML_FP16_TO_FP32(blk->norm);
        float res_norm = GGML_FP16_TO_FP32(blk->res_norm);
        uint32_t seed  = blk->rot_seed;

        // --- Step 1: Unpack indices and look up centroids ---
        int indices[QK_TURBO];
        unpack_3bit_indices(blk->qs, indices);

        float y_hat[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat[i] = turbo_centroids_3bit[indices[i]];
        }

        // --- Step 2: Norm correction ---
        float y_hat_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat_norm_sq += y_hat[i] * y_hat[i];
        }
        float y_hat_norm = sqrtf(y_hat_norm_sq);
        if (y_hat_norm > 1e-10f) {
            float inv_y_hat_norm = 1.0f / y_hat_norm;
            for (int i = 0; i < QK_TURBO; i++) {
                y_hat[i] *= inv_y_hat_norm;
            }
        }

        // --- Step 3: Inverse rotation ---
        float x_pq[QK_TURBO];
        apply_rotation_inverse(y_hat, x_pq, seed);

        // Rescale by original norm
        for (int i = 0; i < QK_TURBO; i++) {
            x_pq[i] *= norm;
        }

        // --- Step 4: QJL residual reconstruction ---
        float residual[QK_TURBO];
        qjl_dequantize(blk->signs, res_norm, seed, residual);

        // --- Step 5: Sum PolarQuant + QJL ---
        for (int i = 0; i < QK_TURBO; i++) {
            yb[i] = x_pq[i] + residual[i];
        }
    }
}

// ============================================================================
// turbo3: 2-bit PolarQuant + 1-bit QJL = 3 bits/val
// ============================================================================

void quantize_row_turbo3_ref(const float * GGML_RESTRICT x, block_turbo3 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k / QK_TURBO;

    for (int64_t ib = 0; ib < nb; ib++) {
        const float * xb = x + ib * QK_TURBO;
        block_turbo3 * blk = &y[ib];

        // Compute norm and normalize
        float norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            norm_sq += xb[i] * xb[i];
        }
        float norm = sqrtf(norm_sq);
        float inv_norm = (norm > 1e-30f) ? 1.0f / norm : 0.0f;

        float x_normalized[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            x_normalized[i] = xb[i] * inv_norm;
        }

        // Forward rotation
        uint32_t seed = make_seed(ib);
        float rotated[QK_TURBO];
        apply_rotation_forward(x_normalized, rotated, seed);

        // Quantize to 2-bit centroids
        int indices[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            indices[i] = nearest_index_2bit(rotated[i]);
        }

        // Dequantize in rotated domain with norm correction
        float y_hat[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat[i] = turbo_centroids_2bit[indices[i]];
        }
        float y_hat_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat_norm_sq += y_hat[i] * y_hat[i];
        }
        float y_hat_norm = sqrtf(y_hat_norm_sq);
        if (y_hat_norm > 1e-10f) {
            float inv = 1.0f / y_hat_norm;
            for (int i = 0; i < QK_TURBO; i++) {
                y_hat[i] *= inv;
            }
        }

        // Inverse rotate and rescale
        float x_pq[QK_TURBO];
        apply_rotation_inverse(y_hat, x_pq, seed);
        for (int i = 0; i < QK_TURBO; i++) {
            x_pq[i] *= norm;
        }

        // Residual
        float residual[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            residual[i] = xb[i] - x_pq[i];
        }
        float res_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            res_norm_sq += residual[i] * residual[i];
        }
        float res_norm = sqrtf(res_norm_sq);

        // QJL
        uint8_t qjl_signs[QK_TURBO / 8];
        qjl_quantize(residual, seed, qjl_signs);

        // Pack
        blk->norm     = GGML_FP32_TO_FP16(norm);
        blk->res_norm = GGML_FP32_TO_FP16(res_norm);
        blk->rot_seed = seed;
        pack_2bit_indices(indices, blk->qs);
        memcpy(blk->signs, qjl_signs, QK_TURBO / 8);
    }
}

void dequantize_row_turbo3(const block_turbo3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k / QK_TURBO;

    for (int64_t ib = 0; ib < nb; ib++) {
        const block_turbo3 * blk = &x[ib];
        float * yb = y + ib * QK_TURBO;

        float norm     = GGML_FP16_TO_FP32(blk->norm);
        float res_norm = GGML_FP16_TO_FP32(blk->res_norm);
        uint32_t seed  = blk->rot_seed;

        int indices[QK_TURBO];
        unpack_2bit_indices(blk->qs, indices);

        float y_hat[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat[i] = turbo_centroids_2bit[indices[i]];
        }

        // Norm correction
        float y_hat_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat_norm_sq += y_hat[i] * y_hat[i];
        }
        float y_hat_norm = sqrtf(y_hat_norm_sq);
        if (y_hat_norm > 1e-10f) {
            float inv = 1.0f / y_hat_norm;
            for (int i = 0; i < QK_TURBO; i++) {
                y_hat[i] *= inv;
            }
        }

        float x_pq[QK_TURBO];
        apply_rotation_inverse(y_hat, x_pq, seed);
        for (int i = 0; i < QK_TURBO; i++) {
            x_pq[i] *= norm;
        }

        float residual[QK_TURBO];
        qjl_dequantize(blk->signs, res_norm, seed, residual);

        for (int i = 0; i < QK_TURBO; i++) {
            yb[i] = x_pq[i] + residual[i];
        }
    }
}

// ============================================================================
// turbo2: 1-bit PolarQuant + 1-bit QJL = 2 bits/val
// ============================================================================

void quantize_row_turbo2_ref(const float * GGML_RESTRICT x, block_turbo2 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k / QK_TURBO;

    for (int64_t ib = 0; ib < nb; ib++) {
        const float * xb = x + ib * QK_TURBO;
        block_turbo2 * blk = &y[ib];

        float norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            norm_sq += xb[i] * xb[i];
        }
        float norm = sqrtf(norm_sq);
        float inv_norm = (norm > 1e-30f) ? 1.0f / norm : 0.0f;

        float x_normalized[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            x_normalized[i] = xb[i] * inv_norm;
        }

        uint32_t seed = make_seed(ib);
        float rotated[QK_TURBO];
        apply_rotation_forward(x_normalized, rotated, seed);

        int indices[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            indices[i] = nearest_index_1bit(rotated[i]);
        }

        float y_hat[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat[i] = turbo_centroids_1bit[indices[i]];
        }
        float y_hat_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat_norm_sq += y_hat[i] * y_hat[i];
        }
        float y_hat_norm = sqrtf(y_hat_norm_sq);
        if (y_hat_norm > 1e-10f) {
            float inv = 1.0f / y_hat_norm;
            for (int i = 0; i < QK_TURBO; i++) {
                y_hat[i] *= inv;
            }
        }

        float x_pq[QK_TURBO];
        apply_rotation_inverse(y_hat, x_pq, seed);
        for (int i = 0; i < QK_TURBO; i++) {
            x_pq[i] *= norm;
        }

        float residual[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            residual[i] = xb[i] - x_pq[i];
        }
        float res_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            res_norm_sq += residual[i] * residual[i];
        }
        float res_norm = sqrtf(res_norm_sq);

        uint8_t qjl_signs[QK_TURBO / 8];
        qjl_quantize(residual, seed, qjl_signs);

        blk->norm     = GGML_FP32_TO_FP16(norm);
        blk->res_norm = GGML_FP32_TO_FP16(res_norm);
        blk->rot_seed = seed;
        pack_1bit_indices(indices, blk->qs);
        memcpy(blk->signs, qjl_signs, QK_TURBO / 8);
    }
}

void dequantize_row_turbo2(const block_turbo2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const int64_t nb = k / QK_TURBO;

    for (int64_t ib = 0; ib < nb; ib++) {
        const block_turbo2 * blk = &x[ib];
        float * yb = y + ib * QK_TURBO;

        float norm     = GGML_FP16_TO_FP32(blk->norm);
        float res_norm = GGML_FP16_TO_FP32(blk->res_norm);
        uint32_t seed  = blk->rot_seed;

        int indices[QK_TURBO];
        unpack_1bit_indices(blk->qs, indices);

        float y_hat[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat[i] = turbo_centroids_1bit[indices[i]];
        }

        float y_hat_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            y_hat_norm_sq += y_hat[i] * y_hat[i];
        }
        float y_hat_norm = sqrtf(y_hat_norm_sq);
        if (y_hat_norm > 1e-10f) {
            float inv = 1.0f / y_hat_norm;
            for (int i = 0; i < QK_TURBO; i++) {
                y_hat[i] *= inv;
            }
        }

        float x_pq[QK_TURBO];
        apply_rotation_inverse(y_hat, x_pq, seed);
        for (int i = 0; i < QK_TURBO; i++) {
            x_pq[i] *= norm;
        }

        float residual[QK_TURBO];
        qjl_dequantize(blk->signs, res_norm, seed, residual);

        for (int i = 0; i < QK_TURBO; i++) {
            yb[i] = x_pq[i] + residual[i];
        }
    }
}

// ============================================================================
// Multi-row quantize wrappers (for ggml quantize dispatch)
// ============================================================================

size_t quantize_turbo4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    size_t row_size = (size_t)(n_per_row / QK_TURBO) * sizeof(block_turbo4);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_ref(src + row * n_per_row,
                                (block_turbo4 *)((char *)dst + row * row_size),
                                n_per_row);
    }
    return nrows * row_size;
}

size_t quantize_turbo3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    size_t row_size = (size_t)(n_per_row / QK_TURBO) * sizeof(block_turbo3);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_ref(src + row * n_per_row,
                                (block_turbo3 *)((char *)dst + row * row_size),
                                n_per_row);
    }
    return nrows * row_size;
}

size_t quantize_turbo2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    size_t row_size = (size_t)(n_per_row / QK_TURBO) * sizeof(block_turbo2);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_ref(src + row * n_per_row,
                                (block_turbo2 *)((char *)dst + row * row_size),
                                n_per_row);
    }
    return nrows * row_size;
}

// ============================================================================
// vec_dot: dot product between turbo-quantized and Q8_0-quantized vectors
//
// Strategy: dequantize each turbo block to float[32], then multiply-accumulate
// against Q8_0's int8 values scaled by Q8_0's delta.  QK_TURBO == QK8_0 == 32
// so blocks align 1:1.  This is the simplest correct implementation; SIMD
// optimization can come later.
// ============================================================================

void ggml_vec_dot_turbo4_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void)nrc; (void)bs; (void)bx; (void)by;
    assert(n % QK_TURBO == 0);

    const block_turbo4 * GGML_RESTRICT x = (const block_turbo4 *)vx;
    const block_q8_0   * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb = n / QK_TURBO;

    float sumf = 0.0f;
    float tmp[QK_TURBO];

    for (int ib = 0; ib < nb; ib++) {
        dequantize_row_turbo4(&x[ib], tmp, QK_TURBO);

        const float d_q8 = GGML_FP16_TO_FP32(y[ib].d);
        float dot = 0.0f;
        for (int j = 0; j < QK_TURBO; j++) {
            dot += tmp[j] * (float)y[ib].qs[j];
        }
        sumf += d_q8 * dot;
    }
    *s = sumf;
}

void ggml_vec_dot_turbo3_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void)nrc; (void)bs; (void)bx; (void)by;
    assert(n % QK_TURBO == 0);

    const block_turbo3 * GGML_RESTRICT x = (const block_turbo3 *)vx;
    const block_q8_0   * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb = n / QK_TURBO;

    float sumf = 0.0f;
    float tmp[QK_TURBO];

    for (int ib = 0; ib < nb; ib++) {
        dequantize_row_turbo3(&x[ib], tmp, QK_TURBO);

        const float d_q8 = GGML_FP16_TO_FP32(y[ib].d);
        float dot = 0.0f;
        for (int j = 0; j < QK_TURBO; j++) {
            dot += tmp[j] * (float)y[ib].qs[j];
        }
        sumf += d_q8 * dot;
    }
    *s = sumf;
}

void ggml_vec_dot_turbo2_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void)nrc; (void)bs; (void)bx; (void)by;
    assert(n % QK_TURBO == 0);

    const block_turbo2 * GGML_RESTRICT x = (const block_turbo2 *)vx;
    const block_q8_0   * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb = n / QK_TURBO;

    float sumf = 0.0f;
    float tmp[QK_TURBO];

    for (int ib = 0; ib < nb; ib++) {
        dequantize_row_turbo2(&x[ib], tmp, QK_TURBO);

        const float d_q8 = GGML_FP16_TO_FP32(y[ib].d);
        float dot = 0.0f;
        for (int j = 0; j < QK_TURBO; j++) {
            dot += tmp[j] * (float)y[ib].qs[j];
        }
        sumf += d_q8 * dot;
    }
    *s = sumf;
}
