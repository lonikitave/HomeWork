/**
 * executor_recursive.c — Recursive Cooley-Tukey FFT executor (scalar C).
 *
 * This is the Phase 0 scalar reference executor.  It calls the handwritten
 * radix-2 butterfly kernel iteratively to implement the full FFT.
 *
 * The RVV-accelerated executor (which dispatches to generated codelets via the
 * plan tree) will be implemented in Phase 3.  This file remains as the
 * correctness baseline and fallback for unsupported transform sizes.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rvvfft.h"
#include "rvvfft_types.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* Forward declaration of the handwritten butterfly */
extern void rvvfft_ref_butterfly_r2_f32_split(
    const float *in_r, const float *in_i,
    float *out_r, float *out_i,
    size_t n,
    const float *tw_r, const float *tw_i);

/* Access plan internals (mirror of struct in planner.c) */
struct rvvfft_plan_s {
    int   n;
    int   sign;
    int   layout;
    int   dtype;
    unsigned flags;
    void *in;
    void *in_i;
    void *out;
    void *out_i;
    void *root;
    float *tw_r;
    float *tw_i;
};

/**
 * rvvfft_execute_recursive — Scalar Cooley-Tukey radix-2 DIT FFT.
 *
 * Operates on split-layout internal buffers.  Interleaved input is first
 * de-interleaved, then the FFT is computed in-place, then re-interleaved.
 */
void rvvfft_execute_recursive(const rvvfft_plan plan)
{
    const struct rvvfft_plan_s *p = (const struct rvvfft_plan_s *)plan;
    int n = p->n;
    int logn = rvvfft_ilog2(n);

    /* Allocate working buffers (split layout) */
    float *buf_r = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                           (size_t)n * sizeof(float));
    float *buf_i = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                           (size_t)n * sizeof(float));
    float *tw_r  = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                           (size_t)(n / 2) * sizeof(float));
    float *tw_i  = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                           (size_t)(n / 2) * sizeof(float));

    if (!buf_r || !buf_i || !tw_r || !tw_i) {
        free(buf_r); free(buf_i); free(tw_r); free(tw_i);
        return;
    }

    /* ── Copy input into split working buffer ──────────────────────── */
    if (p->layout == 0 /* INTERLEAVED */) {
        const float *in = (const float *)p->in;
        for (int k = 0; k < n; ++k) {
            buf_r[k] = in[2 * k];
            buf_i[k] = in[2 * k + 1];
        }
    } else {
        const float *in_r = (const float *)p->in;
        const float *in_i = (const float *)p->in_i;
        memcpy(buf_r, in_r, (size_t)n * sizeof(float));
        memcpy(buf_i, in_i, (size_t)n * sizeof(float));
    }

    /* ── Bit-reversal permutation ──────────────────────────────────── */
    extern void rvvfft_bitrev_inplace_f32_split(float *, float *, int);
    rvvfft_bitrev_inplace_f32_split(buf_r, buf_i, n);

    /* ── Cooley-Tukey DIT butterfly stages ────────────────────────── */
    extern void rvvfft_twiddle_init_f32(float *, float *, int, int, int);

    for (int stage = 0; stage < logn; ++stage) {
        int half  = 1 << stage;           /* half-butterfly size      */
        int span  = half << 1;            /* full butterfly span       */
        int groups = n / span;            /* number of butterfly groups */

        /* Fill twiddle factors for this stage */
        /* W^k_{span} for k = 0 .. half-1 */
        rvvfft_twiddle_init_f32(tw_r, tw_i, span, 1, half);

        /* For BACKWARD transform conjugate the twiddles (negate sin part) */
        if (p->sign == 1 /* RVVFFT_BACKWARD */) {
            for (int k = 0; k < half; ++k)
                tw_i[k] = -tw_i[k];
        }

        /* Apply butterflies across all groups in this stage */
        for (int g = 0; g < groups; ++g) {
            int base = g * span;
            rvvfft_ref_butterfly_r2_f32_split(
                buf_r + base, buf_i + base,
                buf_r + base, buf_i + base,
                (size_t)half,
                tw_r, tw_i);
        }
    }

    /* ── Scale for inverse (1/N) ───────────────────────────────────── */
    if (p->sign == 1 /* RVVFFT_BACKWARD */) {
        float scale = 1.0f / (float)n;
        for (int k = 0; k < n; ++k) {
            buf_r[k] *= scale;
            buf_i[k] *= scale;
        }
    }

    /* ── Copy results to output ────────────────────────────────────── */
    if (p->layout == 0 /* INTERLEAVED */) {
        float *out = (float *)p->out;
        for (int k = 0; k < n; ++k) {
            out[2 * k]     = buf_r[k];
            out[2 * k + 1] = buf_i[k];
        }
    } else {
        float *out_r = (float *)p->out;
        float *out_i = (float *)p->out_i;
        memcpy(out_r, buf_r, (size_t)n * sizeof(float));
        memcpy(out_i, buf_i, (size_t)n * sizeof(float));
    }

    free(buf_r); free(buf_i); free(tw_r); free(tw_i);
}
