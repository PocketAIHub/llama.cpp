// mtmd-swift.h — C-only subset of mtmd.h for Swift interop
// This header strips out C++ includes and wrappers that Swift cannot import.

#ifndef MTMD_SWIFT_H
#define MTMD_SWIFT_H

#include "ggml.h"
#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef MTMD_API
#define MTMD_API
#endif

enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
};

// opaque types
typedef struct mtmd_context      mtmd_context;
typedef struct mtmd_bitmap       mtmd_bitmap;
typedef struct mtmd_image_tokens mtmd_image_tokens;
typedef struct mtmd_input_chunk  mtmd_input_chunk;
typedef struct mtmd_input_chunks mtmd_input_chunks;

struct mtmd_input_text {
    const char * text;
    bool add_special;
    bool parse_special;
};
typedef struct mtmd_input_text mtmd_input_text;

struct mtmd_context_params {
    bool use_gpu;
    bool print_timings;
    int n_threads;
    const char * image_marker;
    const char * media_marker;
    enum llama_flash_attn_type flash_attn_type;
    bool warmup;
    int image_min_tokens;
    int image_max_tokens;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

MTMD_API const char * mtmd_default_marker(void);
MTMD_API struct mtmd_context_params mtmd_context_params_default(void);

MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                            const struct llama_model * text_model,
                                            const struct mtmd_context_params ctx_params);
MTMD_API void mtmd_free(mtmd_context * ctx);

MTMD_API bool mtmd_support_vision(mtmd_context * ctx);

// bitmap
MTMD_API mtmd_bitmap * mtmd_bitmap_init(uint32_t nx, uint32_t ny, const unsigned char * data);
MTMD_API void mtmd_bitmap_free(mtmd_bitmap * bitmap);

// input chunks
MTMD_API mtmd_input_chunks * mtmd_input_chunks_init(void);
MTMD_API size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
MTMD_API const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx);
MTMD_API void mtmd_input_chunks_free(mtmd_input_chunks * chunks);

// input chunk
MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk);
MTMD_API const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output);
MTMD_API const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk);
MTMD_API size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk);

// tokenize
MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
                               mtmd_input_chunks * output,
                               const mtmd_input_text * text,
                               const mtmd_bitmap ** bitmaps,
                               size_t n_bitmaps);

// encode
MTMD_API int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk);
MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);

#endif // MTMD_SWIFT_H
