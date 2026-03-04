/**
 * radix2_f32.c — Handwritten radix-2 butterfly reference kernel (f32, split layout).
 *
 * This file serves as the golden reference against which generated kernels
 * are compared bit-for-bit in CI.  It is intentionally written in portable
 * scalar C so that its correctness can be verified on any host.
 *
 * For production use, prefer the RVV-accelerated generated kernels in
 * src/kernel/generated/.
 */

#include <stddef.h>
#include "rvvfft_types.h"

/**
 * rvvfft_ref_butterfly_r2_f32_split
 *
 * Compute n radix-2 butterflies on split-layout f32 data:
 *
 *   out_r[0..n-1] = in_r[0..n-1] + tw_r * in_r[n..2n-1] - tw_i * in_i[n..2n-1]
 *   out_i[0..n-1] = in_i[0..n-1] + tw_r * in_i[n..2n-1] + tw_i * in_r[n..2n-1]
 *   out_r[n..2n-1] = in_r[0..n-1] - (tw_r * in_r[n..2n-1] - tw_i * in_i[n..2n-1])
 *   out_i[n..2n-1] = in_i[0..n-1] - (tw_r * in_i[n..2n-1] + tw_i * in_r[n..2n-1])
 *
 * @in_r:  Input  real  parts (2*n floats: [top half][bottom half])
 * @in_i:  Input  imag  parts (2*n floats)
 * @out_r: Output real  parts (2*n floats, may alias in_r)
 * @out_i: Output imag  parts (2*n floats, may alias in_i)
 * @n:     Number of butterfly pairs (i.e. half the transform length).
 * @tw_r:  Twiddle real  parts (n floats): cos(-2π k/N)
 * @tw_i:  Twiddle imag  parts (n floats): sin(-2π k/N)
 */
void rvvfft_ref_butterfly_r2_f32_split(
    const float * __restrict__ in_r,
    const float * __restrict__ in_i,
          float * __restrict__ out_r,
          float * __restrict__ out_i,
    size_t n,
    const float * __restrict__ tw_r,
    const float * __restrict__ tw_i)
{
    const float *top_r = in_r;
    const float *top_i = in_i;
    const float *bot_r = in_r + n;
    const float *bot_i = in_i + n;

    float *dst_top_r = out_r;
    float *dst_top_i = out_i;
    float *dst_bot_r = out_r + n;
    float *dst_bot_i = out_i + n;

    for (size_t k = 0; k < n; ++k) {
        /* Complex multiply: (bot_r + j*bot_i) * (tw_r + j*tw_i) */
        float t_r = bot_r[k] * tw_r[k] - bot_i[k] * tw_i[k];
        float t_i = bot_r[k] * tw_i[k] + bot_i[k] * tw_r[k];

        /* Save inputs before overwriting (safe for in-place: in == out) */
        float ar = top_r[k];
        float ai = top_i[k];

        /* Butterfly */
        dst_top_r[k] = ar + t_r;
        dst_top_i[k] = ai + t_i;
        dst_bot_r[k] = ar - t_r;
        dst_bot_i[k] = ai - t_i;
    }
}

/**
 * rvvfft_ref_butterfly_r2_f32_interleaved
 *
 * Interleaved-layout version.  Input/output format: [r0,i0,r1,i1,...].
 *
 * @in:  2*n interleaved complex samples (top n followed by bottom n).
 * @out: 2*n interleaved complex samples (may alias in).
 * @n:   Number of butterfly pairs.
 * @tw_r, @tw_i: Twiddle factor tables as above.
 */
void rvvfft_ref_butterfly_r2_f32_interleaved(
    const float * __restrict__ in,
          float * __restrict__ out,
    size_t n,
    const float * __restrict__ tw_r,
    const float * __restrict__ tw_i)
{
    /* Interleaved: element k has real at [2k], imag at [2k+1] */
    /* Bottom half starts at index 2*n (= n complex samples offset) */
    const float *top = in;
    const float *bot = in + 2 * n;
    float *dst_top   = out;
    float *dst_bot   = out + 2 * n;

    for (size_t k = 0; k < n; ++k) {
        float ar = top[2 * k];
        float ai = top[2 * k + 1];
        float br = bot[2 * k];
        float bi = bot[2 * k + 1];

        float t_r = br * tw_r[k] - bi * tw_i[k];
        float t_i = br * tw_i[k] + bi * tw_r[k];

        dst_top[2 * k]     = ar + t_r;
        dst_top[2 * k + 1] = ai + t_i;
        dst_bot[2 * k]     = ar - t_r;
        dst_bot[2 * k + 1] = ai - t_i;
    }
}
