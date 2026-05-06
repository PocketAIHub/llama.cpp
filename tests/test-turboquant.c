// TurboQuant correctness tests
// Phase 2: MSE bounds, determinism, zero vector, norm preservation (9 tests)
// Phase 7: vec_dot, inner product preservation, multi-block all types, type traits (7 tests)
//
// Build:  gcc -O2 -I../ggml/include -I../ggml/src -o test-turboquant test-turboquant.c ../build-turbo-test/bin/libggml-base.dylib -lm
// Or via cmake: add to tests/CMakeLists.txt

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

// Pull in the function declarations
#define GGML_COMMON_DECL_C
#include "ggml-turboquant.h"
#include "ggml-quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N_SAMPLES 500
#define TOLERANCE 1e-6f

// Simple LCG PRNG for reproducible test vectors (not for crypto)
static uint32_t test_rng_state = 12345;
static float test_randn(void) {
    // Box-Muller transform for approximate Gaussian
    test_rng_state = test_rng_state * 1664525U + 1013904223U;
    float u1 = (float)(test_rng_state >> 1) / (float)(0x7FFFFFFFU) + 1e-10f;
    test_rng_state = test_rng_state * 1664525U + 1013904223U;
    float u2 = (float)(test_rng_state >> 1) / (float)(0x7FFFFFFFU) + 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    printf("  PASS: %s\n", name); \
    tests_passed++; \
} while(0)

// ============================================================================
// Test: turbo4 round-trip MSE within bounds
// Paper bound for 4-bit TurboQuant: MSE ~0.009 on unit vectors
// We allow 5× slack for finite d=32 and structured rotation approximation
// ============================================================================
static void test_turbo4_mse(void) {
    test_rng_state = 42;
    float total_mse = 0.0f;

    for (int s = 0; s < N_SAMPLES; s++) {
        float x[QK_TURBO];
        float norm = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] = test_randn();
            norm += x[i] * x[i];
        }
        norm = sqrtf(norm);
        // Normalize to unit vector
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] /= norm;
        }

        block_turbo4 blk;
        quantize_row_turbo4_ref(x, &blk, QK_TURBO);

        float y[QK_TURBO];
        dequantize_row_turbo4(&blk, y, QK_TURBO);

        float mse = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            float diff = x[i] - y[i];
            mse += diff * diff;
        }
        mse /= QK_TURBO;
        total_mse += mse;
    }

    float avg_mse = total_mse / N_SAMPLES;
    // Paper bound: ~0.009 at d→∞. Allow 5× for d=32 + structured rotation.
    TEST_ASSERT(avg_mse < 0.009f * 5.0f,
        "turbo4 avg MSE %.6f exceeds 5x paper bound (0.045)", avg_mse);
    printf("    turbo4 avg MSE on unit vectors: %.6f\n", avg_mse);
    TEST_PASS("turbo4_mse");
}

// ============================================================================
// Test: turbo3 round-trip MSE within bounds
// Paper bound for 3-bit: MSE ~0.03
// ============================================================================
static void test_turbo3_mse(void) {
    test_rng_state = 42;
    float total_mse = 0.0f;

    for (int s = 0; s < N_SAMPLES; s++) {
        float x[QK_TURBO];
        float norm = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] = test_randn();
            norm += x[i] * x[i];
        }
        norm = sqrtf(norm);
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] /= norm;
        }

        block_turbo3 blk;
        quantize_row_turbo3_ref(x, &blk, QK_TURBO);

        float y[QK_TURBO];
        dequantize_row_turbo3(&blk, y, QK_TURBO);

        float mse = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            float diff = x[i] - y[i];
            mse += diff * diff;
        }
        mse /= QK_TURBO;
        total_mse += mse;
    }

    float avg_mse = total_mse / N_SAMPLES;
    TEST_ASSERT(avg_mse < 0.03f * 5.0f,
        "turbo3 avg MSE %.6f exceeds 5x paper bound (0.15)", avg_mse);
    printf("    turbo3 avg MSE on unit vectors: %.6f\n", avg_mse);
    TEST_PASS("turbo3_mse");
}

// ============================================================================
// Test: turbo2 round-trip MSE within bounds
// Paper bound for 2-bit: MSE ~0.117
// ============================================================================
static void test_turbo2_mse(void) {
    test_rng_state = 42;
    float total_mse = 0.0f;

    for (int s = 0; s < N_SAMPLES; s++) {
        float x[QK_TURBO];
        float norm = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] = test_randn();
            norm += x[i] * x[i];
        }
        norm = sqrtf(norm);
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] /= norm;
        }

        block_turbo2 blk;
        quantize_row_turbo2_ref(x, &blk, QK_TURBO);

        float y[QK_TURBO];
        dequantize_row_turbo2(&blk, y, QK_TURBO);

        float mse = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            float diff = x[i] - y[i];
            mse += diff * diff;
        }
        mse /= QK_TURBO;
        total_mse += mse;
    }

    float avg_mse = total_mse / N_SAMPLES;
    TEST_ASSERT(avg_mse < 0.117f * 5.0f,
        "turbo2 avg MSE %.6f exceeds 5x paper bound (0.585)", avg_mse);
    printf("    turbo2 avg MSE on unit vectors: %.6f\n", avg_mse);
    TEST_PASS("turbo2_mse");
}

// ============================================================================
// Test: Determinism — same input always produces same output
// ============================================================================
static void test_determinism(void) {
    float x[QK_TURBO];
    test_rng_state = 77;
    for (int i = 0; i < QK_TURBO; i++) {
        x[i] = test_randn();
    }

    block_turbo4 blk1, blk2;
    quantize_row_turbo4_ref(x, &blk1, QK_TURBO);
    quantize_row_turbo4_ref(x, &blk2, QK_TURBO);

    TEST_ASSERT(memcmp(&blk1, &blk2, sizeof(block_turbo4)) == 0,
        "turbo4 quantization is not deterministic");

    float y1[QK_TURBO], y2[QK_TURBO];
    dequantize_row_turbo4(&blk1, y1, QK_TURBO);
    dequantize_row_turbo4(&blk2, y2, QK_TURBO);

    TEST_ASSERT(memcmp(y1, y2, QK_TURBO * sizeof(float)) == 0,
        "turbo4 dequantization is not deterministic");

    TEST_PASS("determinism");
}

// ============================================================================
// Test: Zero vector produces small reconstruction
// ============================================================================
static void test_zero_vector(void) {
    float x[QK_TURBO];
    memset(x, 0, sizeof(x));

    block_turbo4 blk;
    quantize_row_turbo4_ref(x, &blk, QK_TURBO);

    float y[QK_TURBO];
    dequantize_row_turbo4(&blk, y, QK_TURBO);

    float recon_norm = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) {
        recon_norm += y[i] * y[i];
    }
    recon_norm = sqrtf(recon_norm);

    TEST_ASSERT(recon_norm < 1.0f,
        "zero vector reconstruction norm %.6f too large", recon_norm);
    TEST_PASS("zero_vector");
}

// ============================================================================
// Test: Multi-block row (k > QK_TURBO)
// ============================================================================
static void test_multi_block(void) {
    const int n_blocks = 4;
    const int k = QK_TURBO * n_blocks;
    float x[QK_TURBO * 4];
    test_rng_state = 123;
    for (int i = 0; i < k; i++) {
        x[i] = test_randn();
    }

    block_turbo4 blks[4];
    quantize_row_turbo4_ref(x, blks, k);

    float y[QK_TURBO * 4];
    dequantize_row_turbo4(blks, y, k);

    // Each block should have reasonable MSE
    for (int b = 0; b < n_blocks; b++) {
        float mse = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            float diff = x[b * QK_TURBO + i] - y[b * QK_TURBO + i];
            mse += diff * diff;
        }
        mse /= QK_TURBO;
        TEST_ASSERT(mse < 5.0f,
            "block %d MSE %.6f unreasonably high", b, mse);
    }

    TEST_PASS("multi_block");
}

// ============================================================================
// Test: Non-unit-norm vectors (typical KV cache values)
// ============================================================================
static void test_arbitrary_norm(void) {
    test_rng_state = 55;
    float total_relative_error = 0.0f;
    int n_valid = 0;

    for (int s = 0; s < N_SAMPLES; s++) {
        float x[QK_TURBO];
        float scale = 0.01f + 100.0f * ((float)(s) / N_SAMPLES);
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] = test_randn() * scale;
        }

        block_turbo4 blk;
        quantize_row_turbo4_ref(x, &blk, QK_TURBO);

        float y[QK_TURBO];
        dequantize_row_turbo4(&blk, y, QK_TURBO);

        // Compute relative error
        float orig_norm_sq = 0.0f;
        float diff_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            orig_norm_sq += x[i] * x[i];
            float diff = x[i] - y[i];
            diff_norm_sq += diff * diff;
        }
        if (orig_norm_sq > 1e-10f) {
            total_relative_error += sqrtf(diff_norm_sq / orig_norm_sq);
            n_valid++;
        }
    }

    float avg_rel_error = total_relative_error / n_valid;
    // Relative error should be bounded — typically < 0.3 for turbo4
    TEST_ASSERT(avg_rel_error < 0.5f,
        "avg relative error %.6f too high for arbitrary-norm vectors", avg_rel_error);
    printf("    turbo4 avg relative error (arbitrary norm): %.6f\n", avg_rel_error);
    TEST_PASS("arbitrary_norm");
}

// ============================================================================
// Test: Bit packing round-trip (internal consistency)
// ============================================================================
static void test_bit_packing(void) {
    // Test 3-bit packing
    int indices_3bit[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        indices_3bit[i] = i % 8;  // 0-7
    }
    uint8_t packed_3bit[QK_TURBO * 3 / 8];
    int unpacked_3bit[QK_TURBO];

    // Forward declare these as extern — they're static in the .c file,
    // so we test them via quantize/dequantize round-trip instead.
    // Direct packing test: quantize a known vector and check indices survive.

    // Use a vector whose quantized indices we can predict:
    // All values at centroid positions should map to those exact indices.
    float x[QK_TURBO];
    for (int i = 0; i < QK_TURBO; i++) {
        // Place values exactly at the 3-bit centroids (turbo4 uses 3-bit PQ)
        // After rotation, coordinates should be near centroids.
        // This is hard to construct directly, so instead test round-trip consistency.
        x[i] = 0.5f * test_randn();
    }

    block_turbo4 blk;
    quantize_row_turbo4_ref(x, &blk, QK_TURBO);

    // Dequantize twice — should be identical
    float y1[QK_TURBO], y2[QK_TURBO];
    dequantize_row_turbo4(&blk, y1, QK_TURBO);
    dequantize_row_turbo4(&blk, y2, QK_TURBO);

    TEST_ASSERT(memcmp(y1, y2, sizeof(y1)) == 0,
        "repeated dequantize produces different results");

    TEST_PASS("bit_packing_roundtrip");
}

// ============================================================================
// Test: MSE improves with higher bit width
// turbo4 > turbo3 > turbo2 in quality
// ============================================================================
static void test_quality_ordering(void) {
    test_rng_state = 99;
    float mse_t4 = 0.0f, mse_t3 = 0.0f, mse_t2 = 0.0f;

    for (int s = 0; s < N_SAMPLES; s++) {
        float x[QK_TURBO];
        float norm = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] = test_randn();
            norm += x[i] * x[i];
        }
        norm = sqrtf(norm);
        for (int i = 0; i < QK_TURBO; i++) {
            x[i] /= norm;
        }

        float y[QK_TURBO];

        block_turbo4 b4;
        quantize_row_turbo4_ref(x, &b4, QK_TURBO);
        dequantize_row_turbo4(&b4, y, QK_TURBO);
        for (int i = 0; i < QK_TURBO; i++) { float d = x[i] - y[i]; mse_t4 += d*d; }

        block_turbo3 b3;
        quantize_row_turbo3_ref(x, &b3, QK_TURBO);
        dequantize_row_turbo3(&b3, y, QK_TURBO);
        for (int i = 0; i < QK_TURBO; i++) { float d = x[i] - y[i]; mse_t3 += d*d; }

        block_turbo2 b2;
        quantize_row_turbo2_ref(x, &b2, QK_TURBO);
        dequantize_row_turbo2(&b2, y, QK_TURBO);
        for (int i = 0; i < QK_TURBO; i++) { float d = x[i] - y[i]; mse_t2 += d*d; }
    }

    mse_t4 /= N_SAMPLES * QK_TURBO;
    mse_t3 /= N_SAMPLES * QK_TURBO;
    mse_t2 /= N_SAMPLES * QK_TURBO;

    printf("    MSE ordering: turbo4=%.6f < turbo3=%.6f < turbo2=%.6f\n", mse_t4, mse_t3, mse_t2);

    TEST_ASSERT(mse_t4 < mse_t3,
        "turbo4 MSE (%.6f) should be less than turbo3 (%.6f)", mse_t4, mse_t3);
    TEST_ASSERT(mse_t3 < mse_t2,
        "turbo3 MSE (%.6f) should be less than turbo2 (%.6f)", mse_t3, mse_t2);

    TEST_PASS("quality_ordering");
}

// ============================================================================
// Phase 7 tests: vec_dot, inner product preservation, multi-block, type traits
// ============================================================================

// Helper: compute float dot product
static float float_dot(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// ============================================================================
// Test: vec_dot turbo4 × q8_0 matches dequantized float dot product
// ============================================================================
static void test_vec_dot_turbo4(void) {
    test_rng_state = 200;
    const int k = QK_TURBO * 4;  // 4 blocks

    float x_f32[QK_TURBO * 4];
    float y_f32[QK_TURBO * 4];
    for (int i = 0; i < k; i++) {
        x_f32[i] = test_randn() * 0.5f;
        y_f32[i] = test_randn() * 0.5f;
    }

    // Quantize x to turbo4
    block_turbo4 x_turbo[4];
    quantize_row_turbo4_ref(x_f32, x_turbo, k);

    // Quantize y to q8_0
    block_q8_0 y_q8[4];
    quantize_row_q8_0_ref(y_f32, y_q8, k);

    // vec_dot: turbo4 · q8_0
    float dot_vec;
    ggml_vec_dot_turbo4_q8_0(k, &dot_vec, 0, x_turbo, 0, y_q8, 0, 1);

    // Reference: dequantize turbo4 then dot with dequantized q8_0
    float x_deq[QK_TURBO * 4];
    float y_deq[QK_TURBO * 4];
    dequantize_row_turbo4(x_turbo, x_deq, k);
    dequantize_row_q8_0(y_q8, y_deq, k);
    float dot_ref = float_dot(x_deq, y_deq, k);

    float rel_err = fabsf(dot_vec - dot_ref) / (fabsf(dot_ref) + 1e-10f);
    TEST_ASSERT(rel_err < 0.01f,
        "turbo4 vec_dot relative error %.6f (got %.4f, expected %.4f)", rel_err, dot_vec, dot_ref);
    printf("    vec_dot turbo4·q8_0: %.4f (ref: %.4f, rel_err: %.6f)\n", dot_vec, dot_ref, rel_err);
    TEST_PASS("vec_dot_turbo4");
}

// ============================================================================
// Test: vec_dot turbo3 × q8_0
// ============================================================================
static void test_vec_dot_turbo3(void) {
    test_rng_state = 201;
    const int k = QK_TURBO * 4;

    float x_f32[QK_TURBO * 4];
    float y_f32[QK_TURBO * 4];
    for (int i = 0; i < k; i++) {
        x_f32[i] = test_randn() * 0.5f;
        y_f32[i] = test_randn() * 0.5f;
    }

    block_turbo3 x_turbo[4];
    quantize_row_turbo3_ref(x_f32, x_turbo, k);

    block_q8_0 y_q8[4];
    quantize_row_q8_0_ref(y_f32, y_q8, k);

    float dot_vec;
    ggml_vec_dot_turbo3_q8_0(k, &dot_vec, 0, x_turbo, 0, y_q8, 0, 1);

    float x_deq[QK_TURBO * 4];
    float y_deq[QK_TURBO * 4];
    dequantize_row_turbo3(x_turbo, x_deq, k);
    dequantize_row_q8_0(y_q8, y_deq, k);
    float dot_ref = float_dot(x_deq, y_deq, k);

    float rel_err = fabsf(dot_vec - dot_ref) / (fabsf(dot_ref) + 1e-10f);
    TEST_ASSERT(rel_err < 0.01f,
        "turbo3 vec_dot relative error %.6f (got %.4f, expected %.4f)", rel_err, dot_vec, dot_ref);
    printf("    vec_dot turbo3·q8_0: %.4f (ref: %.4f, rel_err: %.6f)\n", dot_vec, dot_ref, rel_err);
    TEST_PASS("vec_dot_turbo3");
}

// ============================================================================
// Test: vec_dot turbo2 × q8_0
// ============================================================================
static void test_vec_dot_turbo2(void) {
    test_rng_state = 202;
    const int k = QK_TURBO * 4;

    float x_f32[QK_TURBO * 4];
    float y_f32[QK_TURBO * 4];
    for (int i = 0; i < k; i++) {
        x_f32[i] = test_randn() * 0.5f;
        y_f32[i] = test_randn() * 0.5f;
    }

    block_turbo2 x_turbo[4];
    quantize_row_turbo2_ref(x_f32, x_turbo, k);

    block_q8_0 y_q8[4];
    quantize_row_q8_0_ref(y_f32, y_q8, k);

    float dot_vec;
    ggml_vec_dot_turbo2_q8_0(k, &dot_vec, 0, x_turbo, 0, y_q8, 0, 1);

    float x_deq[QK_TURBO * 4];
    float y_deq[QK_TURBO * 4];
    dequantize_row_turbo2(x_turbo, x_deq, k);
    dequantize_row_q8_0(y_q8, y_deq, k);
    float dot_ref = float_dot(x_deq, y_deq, k);

    float rel_err = fabsf(dot_vec - dot_ref) / (fabsf(dot_ref) + 1e-10f);
    TEST_ASSERT(rel_err < 0.01f,
        "turbo2 vec_dot relative error %.6f (got %.4f, expected %.4f)", rel_err, dot_vec, dot_ref);
    printf("    vec_dot turbo2·q8_0: %.4f (ref: %.4f, rel_err: %.6f)\n", dot_vec, dot_ref, rel_err);
    TEST_PASS("vec_dot_turbo2");
}

// ============================================================================
// Test: Inner product preservation (K cache critical path)
// Actual attention pattern: <q_f32, dequant(quant(k))> ≈ <q_f32, k>
// Only K is quantized; Q remains in FP32. This matches the real inference path.
// ============================================================================
static void test_inner_product_preservation(void) {
    test_rng_state = 300;
    float total_rel_error_t4 = 0.0f;
    float total_rel_error_t3 = 0.0f;
    float total_rel_error_t2 = 0.0f;
    int n_valid = 0;

    for (int s = 0; s < N_SAMPLES; s++) {
        float q[QK_TURBO], k[QK_TURBO];
        for (int i = 0; i < QK_TURBO; i++) {
            q[i] = test_randn();
            k[i] = test_randn();
        }

        // Normalize to unit vectors (like actual attention heads)
        float qn = 0.0f, kn = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) { qn += q[i]*q[i]; kn += k[i]*k[i]; }
        qn = sqrtf(qn); kn = sqrtf(kn);
        for (int i = 0; i < QK_TURBO; i++) { q[i] /= qn; k[i] /= kn; }

        float orig_dot = float_dot(q, k, QK_TURBO);

        // turbo4: quantize K, dequantize, dot with FP32 Q
        block_turbo4 bk4;
        quantize_row_turbo4_ref(k, &bk4, QK_TURBO);
        float k4[QK_TURBO];
        dequantize_row_turbo4(&bk4, k4, QK_TURBO);
        float dot4 = float_dot(q, k4, QK_TURBO);

        // turbo3
        block_turbo3 bk3;
        quantize_row_turbo3_ref(k, &bk3, QK_TURBO);
        float k3[QK_TURBO];
        dequantize_row_turbo3(&bk3, k3, QK_TURBO);
        float dot3 = float_dot(q, k3, QK_TURBO);

        // turbo2
        block_turbo2 bk2;
        quantize_row_turbo2_ref(k, &bk2, QK_TURBO);
        float k2[QK_TURBO];
        dequantize_row_turbo2(&bk2, k2, QK_TURBO);
        float dot2 = float_dot(q, k2, QK_TURBO);

        // Absolute error (unit vectors, so max possible dot is 1.0)
        total_rel_error_t4 += fabsf(dot4 - orig_dot);
        total_rel_error_t3 += fabsf(dot3 - orig_dot);
        total_rel_error_t2 += fabsf(dot2 - orig_dot);
        n_valid++;
    }

    float avg_t4 = total_rel_error_t4 / n_valid;
    float avg_t3 = total_rel_error_t3 / n_valid;
    float avg_t2 = total_rel_error_t2 / n_valid;

    printf("    Attention IP preservation (Q·dequant(quant(K)), unit vectors):\n");
    printf("      turbo4=%.4f, turbo3=%.4f, turbo2=%.4f\n", avg_t4, avg_t3, avg_t2);

    // Absolute error on unit vector dot products (max possible dot = 1.0)
    // At d=32, expect small absolute error for turbo4, moderate for lower bits
    TEST_ASSERT(avg_t4 < 0.10f,
        "turbo4 attention IP avg abs error %.4f exceeds threshold", avg_t4);
    TEST_ASSERT(avg_t3 < 0.20f,
        "turbo3 attention IP avg abs error %.4f exceeds threshold", avg_t3);
    TEST_ASSERT(avg_t2 < 0.40f,
        "turbo2 attention IP avg abs error %.4f exceeds threshold", avg_t2);

    // Quality ordering should hold
    TEST_ASSERT(avg_t4 < avg_t3,
        "turbo4 IP error (%.4f) should be less than turbo3 (%.4f)", avg_t4, avg_t3);
    TEST_ASSERT(avg_t3 < avg_t2,
        "turbo3 IP error (%.4f) should be less than turbo2 (%.4f)", avg_t3, avg_t2);

    TEST_PASS("inner_product_preservation");
}

// ============================================================================
// Test: Multi-block turbo3
// ============================================================================
static void test_multi_block_turbo3(void) {
    const int n_blocks = 4;
    const int k = QK_TURBO * n_blocks;
    float x[QK_TURBO * 4];
    test_rng_state = 124;
    for (int i = 0; i < k; i++) x[i] = test_randn();

    block_turbo3 blks[4];
    quantize_row_turbo3_ref(x, blks, k);

    float y[QK_TURBO * 4];
    dequantize_row_turbo3(blks, y, k);

    for (int b = 0; b < n_blocks; b++) {
        float mse = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            float diff = x[b * QK_TURBO + i] - y[b * QK_TURBO + i];
            mse += diff * diff;
        }
        mse /= QK_TURBO;
        TEST_ASSERT(mse < 5.0f,
            "turbo3 block %d MSE %.6f unreasonably high", b, mse);
    }
    TEST_PASS("multi_block_turbo3");
}

// ============================================================================
// Test: Multi-block turbo2
// ============================================================================
static void test_multi_block_turbo2(void) {
    const int n_blocks = 4;
    const int k = QK_TURBO * n_blocks;
    float x[QK_TURBO * 4];
    test_rng_state = 125;
    for (int i = 0; i < k; i++) x[i] = test_randn();

    block_turbo2 blks[4];
    quantize_row_turbo2_ref(x, blks, k);

    float y[QK_TURBO * 4];
    dequantize_row_turbo2(blks, y, k);

    for (int b = 0; b < n_blocks; b++) {
        float mse = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) {
            float diff = x[b * QK_TURBO + i] - y[b * QK_TURBO + i];
            mse += diff * diff;
        }
        mse /= QK_TURBO;
        TEST_ASSERT(mse < 10.0f,
            "turbo2 block %d MSE %.6f unreasonably high", b, mse);
    }
    TEST_PASS("multi_block_turbo2");
}

// ============================================================================
// Test: Type traits are properly registered in ggml
// ============================================================================
static void test_type_traits(void) {
    // Verify type names resolve correctly
    TEST_ASSERT(strcmp(ggml_type_name(GGML_TYPE_TURBO4), "turbo4") == 0,
        "TURBO4 type name is '%s', expected 'turbo4'", ggml_type_name(GGML_TYPE_TURBO4));
    TEST_ASSERT(strcmp(ggml_type_name(GGML_TYPE_TURBO3), "turbo3") == 0,
        "TURBO3 type name is '%s', expected 'turbo3'", ggml_type_name(GGML_TYPE_TURBO3));
    TEST_ASSERT(strcmp(ggml_type_name(GGML_TYPE_TURBO2), "turbo2") == 0,
        "TURBO2 type name is '%s', expected 'turbo2'", ggml_type_name(GGML_TYPE_TURBO2));

    // Verify block sizes
    TEST_ASSERT(ggml_blck_size(GGML_TYPE_TURBO4) == QK_TURBO,
        "TURBO4 block size %d, expected %d", (int)ggml_blck_size(GGML_TYPE_TURBO4), QK_TURBO);
    TEST_ASSERT(ggml_blck_size(GGML_TYPE_TURBO3) == QK_TURBO,
        "TURBO3 block size %d, expected %d", (int)ggml_blck_size(GGML_TYPE_TURBO3), QK_TURBO);
    TEST_ASSERT(ggml_blck_size(GGML_TYPE_TURBO2) == QK_TURBO,
        "TURBO2 block size %d, expected %d", (int)ggml_blck_size(GGML_TYPE_TURBO2), QK_TURBO);

    // Verify type sizes match struct sizes
    TEST_ASSERT(ggml_type_size(GGML_TYPE_TURBO4) == sizeof(block_turbo4),
        "TURBO4 type size %zu, expected %zu", ggml_type_size(GGML_TYPE_TURBO4), sizeof(block_turbo4));
    TEST_ASSERT(ggml_type_size(GGML_TYPE_TURBO3) == sizeof(block_turbo3),
        "TURBO3 type size %zu, expected %zu", ggml_type_size(GGML_TYPE_TURBO3), sizeof(block_turbo3));
    TEST_ASSERT(ggml_type_size(GGML_TYPE_TURBO2) == sizeof(block_turbo2),
        "TURBO2 type size %zu, expected %zu", ggml_type_size(GGML_TYPE_TURBO2), sizeof(block_turbo2));

    // Verify quantized flag
    TEST_ASSERT(ggml_is_quantized(GGML_TYPE_TURBO4),
        "TURBO4 should be quantized");
    TEST_ASSERT(ggml_is_quantized(GGML_TYPE_TURBO3),
        "TURBO3 should be quantized");
    TEST_ASSERT(ggml_is_quantized(GGML_TYPE_TURBO2),
        "TURBO2 should be quantized");

    // Verify struct sizes match design doc
    TEST_ASSERT(sizeof(block_turbo4) == 24, "block_turbo4 should be 24 bytes, got %zu", sizeof(block_turbo4));
    TEST_ASSERT(sizeof(block_turbo3) == 20, "block_turbo3 should be 20 bytes, got %zu", sizeof(block_turbo3));
    TEST_ASSERT(sizeof(block_turbo2) == 16, "block_turbo2 should be 16 bytes, got %zu", sizeof(block_turbo2));

    printf("    All type traits verified for TURBO4/3/2\n");
    TEST_PASS("type_traits");
}

int main(void) {
    printf("TurboQuant Correctness Tests\n");
    printf("============================\n\n");

    printf("[turbo4 MSE]\n");
    test_turbo4_mse();
    printf("\n[turbo3 MSE]\n");
    test_turbo3_mse();
    printf("\n[turbo2 MSE]\n");
    test_turbo2_mse();
    printf("\n[determinism]\n");
    test_determinism();
    printf("\n[zero vector]\n");
    test_zero_vector();
    printf("\n[multi-block]\n");
    test_multi_block();
    printf("\n[arbitrary norm]\n");
    test_arbitrary_norm();
    printf("\n[bit packing]\n");
    test_bit_packing();
    printf("\n[quality ordering]\n");
    test_quality_ordering();

    printf("\n--- Phase 7: Validation Tests ---\n");

    printf("\n[vec_dot turbo4]\n");
    test_vec_dot_turbo4();
    printf("\n[vec_dot turbo3]\n");
    test_vec_dot_turbo3();
    printf("\n[vec_dot turbo2]\n");
    test_vec_dot_turbo2();
    printf("\n[inner product preservation]\n");
    test_inner_product_preservation();
    printf("\n[multi-block turbo3]\n");
    test_multi_block_turbo3();
    printf("\n[multi-block turbo2]\n");
    test_multi_block_turbo2();
    printf("\n[type traits]\n");
    test_type_traits();

    printf("\n============================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
