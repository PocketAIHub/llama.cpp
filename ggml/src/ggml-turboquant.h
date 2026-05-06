#pragma once

// TurboQuant KV cache compression: PolarQuant + QJL
// Reference: TurboQuant (ICLR 2026, arXiv:2504.19874)

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Row quantize/dequantize (registered in ggml.c type_traits) ---

void quantize_row_turbo4_ref(const float * GGML_RESTRICT x, block_turbo4 * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo4(const block_turbo4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_turbo3_ref(const float * GGML_RESTRICT x, block_turbo3 * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo3(const block_turbo3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_turbo2_ref(const float * GGML_RESTRICT x, block_turbo2 * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo2(const block_turbo2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// --- Multi-row quantize wrappers (used by ggml_quantize_chunk) ---

size_t quantize_turbo4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// --- vec_dot: dot product between turbo-quantized and Q8_0-quantized vectors ---
// Used by CPU backend for attention score computation (Q @ K^T).
// Dequantize-then-dot reference implementation; fused version in Phase 8.

void ggml_vec_dot_turbo4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_turbo3_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_turbo2_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
