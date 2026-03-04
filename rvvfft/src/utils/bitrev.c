/**
 * bitrev.c — Bit-reversal permutation utilities.
 *
 * After a Cooley-Tukey decimation-in-time FFT the output samples are in
 * bit-reversed order.  This module provides in-place bit-reversal functions
 * that can be used either pre- or post-transform.
 */

#include <stddef.h>
#include <stdint.h>
#include "rvvfft_types.h"

/* ── Helpers ───────────────────────────────────────────────────────────────── */

/**
 * rvvfft_bitrev_index — Compute the bit-reversal of index k for log2(n) bits.
 *
 * @k:    Input index  (0 ≤ k < n).
 * @logn: Number of bits = log2(n).
 */
static inline uint32_t bitrev_index(uint32_t k, int logn)
{
    uint32_t rev = 0;
    for (int b = 0; b < logn; ++b) {
        rev = (rev << 1) | (k & 1u);
        k >>= 1;
    }
    return rev;
}

/* ── Single precision, split layout ───────────────────────────────────────── */

/**
 * rvvfft_bitrev_inplace_f32_split — In-place bit-reversal permutation (f32,
 *                                    split layout).
 *
 * @x_r: Real  part array of n elements.
 * @x_i: Imag  part array of n elements.
 * @n:   Transform length (must be a power of 2).
 */
void rvvfft_bitrev_inplace_f32_split(float *x_r, float *x_i, int n)
{
    int logn = rvvfft_ilog2(n);
    for (int k = 1; k < n - 1; ++k) {
        uint32_t rev = bitrev_index((uint32_t)k, logn);
        if (rev > (uint32_t)k) {
            /* Swap complex samples k and rev */
            float tmp_r = x_r[k]; x_r[k] = x_r[rev]; x_r[rev] = tmp_r;
            float tmp_i = x_i[k]; x_i[k] = x_i[rev]; x_i[rev] = tmp_i;
        }
    }
}

/* ── Single precision, interleaved layout ─────────────────────────────────── */

/**
 * rvvfft_bitrev_inplace_f32_interleaved — In-place bit-reversal (f32,
 *                                          interleaved [r0,i0,r1,i1,...]).
 *
 * @x: Interleaved array of 2*n floats.
 * @n: Transform length (power of 2).
 */
void rvvfft_bitrev_inplace_f32_interleaved(float *x, int n)
{
    int logn = rvvfft_ilog2(n);
    for (int k = 1; k < n - 1; ++k) {
        uint32_t rev = bitrev_index((uint32_t)k, logn);
        if (rev > (uint32_t)k) {
            float tmp_r = x[2 * k];     x[2 * k]     = x[2 * rev];     x[2 * rev]     = tmp_r;
            float tmp_i = x[2 * k + 1]; x[2 * k + 1] = x[2 * rev + 1]; x[2 * rev + 1] = tmp_i;
        }
    }
}

/* ── Double precision, split layout ───────────────────────────────────────── */

/**
 * rvvfft_bitrev_inplace_f64_split — In-place bit-reversal (f64, split layout).
 */
void rvvfft_bitrev_inplace_f64_split(double *x_r, double *x_i, int n)
{
    int logn = rvvfft_ilog2(n);
    for (int k = 1; k < n - 1; ++k) {
        uint32_t rev = bitrev_index((uint32_t)k, logn);
        if (rev > (uint32_t)k) {
            double tmp_r = x_r[k]; x_r[k] = x_r[rev]; x_r[rev] = tmp_r;
            double tmp_i = x_i[k]; x_i[k] = x_i[rev]; x_i[rev] = tmp_i;
        }
    }
}
