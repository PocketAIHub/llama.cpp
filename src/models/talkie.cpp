#include "models.h"

#include <cstring>

void llama_model_talkie::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_talkie::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // Upstream-converted GGUFs use attn_q_norm/layer_output_scale. Older
        // PocketAI GGUFs used attn_q_gain/embed_skip; keep those as fallbacks.
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {1, n_head}, TENSOR_NOT_REQUIRED);
        if (!layer.attn_q_norm) {
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_GAIN, "weight", i), {1, n_head}, 0);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1}, TENSOR_NOT_REQUIRED);
        if (!layer.out_scale) {
            layer.out_scale = create_tensor(tn(LLM_TENSOR_EMBED_SKIP, "weight", i), {n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_talkie::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_talkie::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * e_x = build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1);
    e_x = ggml_dup(ctx0, e_x);
    cb(e_x, "embd_norm", -1);

    cur = ggml_dup(ctx0, e_x);

    const bool legacy_tensor_names =
        model.layers[0].attn_q_norm != nullptr &&
        std::strstr(model.layers[0].attn_q_norm->name, "attn_q_gain") != nullptr;
    const float talkie_freq_scale = legacy_tensor_names ? -1.0f * freq_scale : freq_scale;
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * attn_residual = cur;

        ggml_tensor * attn_in = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, il);
        cb(attn_in, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], attn_in,
                n_embd_head, n_head, n_head_kv, il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, talkie_freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, talkie_freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        Qcur = build_norm(Qcur, nullptr, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_norm", il);

        Kcur = build_norm(Kcur, nullptr, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_norm", il);

        ggml_tensor * q_gain_f32 = ggml_cast(ctx0, model.layers[il].attn_q_norm, GGML_TYPE_F32);
        Qcur = ggml_mul(ctx0, Qcur, q_gain_f32);
        cb(Qcur, "Qcur_gained", il);

        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn,
                model.layers[il].wo, nullptr, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        cur = ggml_add(ctx0, cur, attn_residual);
        cb(cur, "after_attn_residual", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            e_x = ggml_get_rows(ctx0, e_x, inp_out_ids);
        }

        ggml_tensor * ffn_residual = cur;

        ggml_tensor * mlp_in = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, il);
        cb(mlp_in, "ffn_norm", il);

        cur = build_ffn(mlp_in,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "after_ffn_residual", il);

        ggml_tensor * skip_scale_f32 = ggml_cast(ctx0, model.layers[il].out_scale, GGML_TYPE_F32);
        ggml_tensor * skip_term = ggml_mul(ctx0, e_x, skip_scale_f32);
        cur = ggml_add(ctx0, cur, skip_term);
        cb(cur, "after_embed_skip", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
