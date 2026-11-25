#pragma once

// moeq binary-factor experts: the op sequence of one expert bank, shared by llm_graph_context::build_factored_mm_id and
// tests/test-factored-experts.cpp so that the test checks the graph the model runs.
//
// Per selected expert e of a token, with x the token's input row:
//     y = s_post[e] * ( sign(U_e) * ( sign(V_e)^T * ( s_pre[e] * x ) ) )
// v      = sign(V)^T, S1_32 [n_in,  r,     n_expert]
// u      = sign(U),   S1_32 [r,     n_out, n_expert]
// s_pre  = F16 [n_in,  1, n_expert]
// s_post = F16 [n_out, 1, n_expert]
// cur    = [n_in, 1 or n_expert_used, n_tokens] (one row per token broadcast to its experts, or one row per slot)
// ids    = I32 [n_expert_used, n_tokens], the selected experts
// The scales of the selected experts are gathered with the ids flattened, so no [n, n_expert, n_tokens] tensor is
// materialized. Only standard ops are used: GET_ROWS, MUL and two MUL_MAT_ID per bank.

#include "ggml.h"

struct llm_factored_mm_id_result {
    ggml_tensor * x_scaled;   // s_pre[e] * x                 [n_in,  n_expert_used, n_tokens]
    ggml_tensor * z;          // sign(V_e)^T x_scaled         [r,     n_expert_used, n_tokens]
    ggml_tensor * y_unscaled; // sign(U_e) z                  [n_out, n_expert_used, n_tokens]
    ggml_tensor * y;          // s_post[e] * y_unscaled       [n_out, n_expert_used, n_tokens]
};

static inline llm_factored_mm_id_result llm_build_factored_mm_id(
        ggml_context * ctx,
        ggml_tensor  * v,
        ggml_tensor  * u,
        ggml_tensor  * s_pre,
        ggml_tensor  * s_post,
        ggml_tensor  * cur,
        ggml_tensor  * ids) {
    const int64_t n_in     = v->ne[0];
    const int64_t rank     = v->ne[1];
    const int64_t n_expert = v->ne[2];
    const int64_t n_out    = u->ne[1];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];

    GGML_ASSERT(u->ne[0] == rank && u->ne[2] == n_expert);
    GGML_ASSERT(s_pre->ne[0]  == n_in  && s_pre->ne[1]  == 1 && s_pre->ne[2]  == n_expert);
    GGML_ASSERT(s_post->ne[0] == n_out && s_post->ne[1] == 1 && s_post->ne[2] == n_expert);
    GGML_ASSERT(cur->ne[0] == n_in && cur->ne[2] == n_tokens && (cur->ne[1] == 1 || cur->ne[1] == n_used));

    ggml_tensor * ids_flat = ggml_reshape_1d(ctx, ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids), n_used*n_tokens);

    ggml_tensor * sp = ggml_get_rows(ctx, ggml_reshape_2d(ctx, s_pre, n_in, n_expert), ids_flat); // [n_in, n_used*n_tokens]
    sp = ggml_reshape_3d(ctx, sp, n_in, n_used, n_tokens);

    llm_factored_mm_id_result res;
    // Operand order fixes where the graph puts the gather: ggml visits a node's first source first, so with the
    // per-slot input first (the down bank: the GLU output) the gather lands right before this MUL, and the CUDA
    // backend computes gather and product in one kernel (mul-gathered.cuh). A broadcast input ([n_in, 1, n_tokens], the
    // gate-up bank) must be the second operand; it is computed earlier anyway (the router reads it), so only views lie
    // between the gather and the MUL. The product is the same either way (one F32 multiply, commutative).
    res.x_scaled   = cur->ne[1] == n_used ? ggml_mul(ctx, cur, sp) : ggml_mul(ctx, sp, cur);
    res.z          = ggml_mul_mat_id(ctx, v, res.x_scaled, ids);
    res.y_unscaled = ggml_mul_mat_id(ctx, u, res.z, ids);

    ggml_tensor * so = ggml_get_rows(ctx, ggml_reshape_2d(ctx, s_post, n_out, n_expert), ids_flat); // [n_out, n_used*n_tokens]
    so = ggml_reshape_3d(ctx, so, n_out, n_used, n_tokens);

    res.y = ggml_mul(ctx, res.y_unscaled, so);

    return res;
}

// The dense form (a feed-forward without experts, e.g. Qwen3.5's): one bank applied to every token,
//     y = s_post * ( sign(U) * ( sign(V)^T * ( s_pre * x ) ) )
// v      = sign(V)^T, S1_32 [n_in,  r]
// u      = sign(U),   S1_32 [r,     n_out]
// s_pre  = F16 or F32 [n_in],  s_post = F16 or F32 [n_out]: cast to F32 in the graph (the CPU's MUL has no F32 x F16
//          kernel; one vector per bank)
// cur    = [n_in, n_tokens]  ->  y [n_out, n_tokens]
// Ops: (CPY), MUL, two MUL_MAT, MUL, so a backend's S1_32 MUL_MAT kernel serves it.
static inline llm_factored_mm_id_result llm_build_factored_mm(
        ggml_context * ctx,
        ggml_tensor  * v,
        ggml_tensor  * u,
        ggml_tensor  * s_pre,
        ggml_tensor  * s_post,
        ggml_tensor  * cur) {
    const int64_t n_in  = v->ne[0];
    const int64_t rank  = v->ne[1];
    const int64_t n_out = u->ne[1];

    GGML_ASSERT(ggml_n_dims(v) == 2 && ggml_n_dims(u) == 2 && u->ne[0] == rank);
    GGML_ASSERT(ggml_nelements(s_pre) == n_in && ggml_nelements(s_post) == n_out);
    GGML_ASSERT(cur->ne[0] == n_in);

    ggml_tensor * sp = s_pre->type  == GGML_TYPE_F32 ? s_pre  : ggml_cast(ctx, s_pre,  GGML_TYPE_F32);
    ggml_tensor * so = s_post->type == GGML_TYPE_F32 ? s_post : ggml_cast(ctx, s_post, GGML_TYPE_F32);

    llm_factored_mm_id_result res;
    res.x_scaled   = ggml_mul(ctx, cur, sp);                // [n_in,  n_tokens]
    res.z          = ggml_mul_mat(ctx, v, res.x_scaled);    // [r,     n_tokens]
    res.y_unscaled = ggml_mul_mat(ctx, u, res.z);           // [n_out, n_tokens]
    res.y          = ggml_mul(ctx, res.y_unscaled, so);

    return res;
}
