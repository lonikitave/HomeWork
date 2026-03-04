/**
 * twiddle.c — Twiddle factor table generation utilities.
 *
 * Twiddle factors for an N-point FFT are the N-th roots of unity:
 *   W^k_N = e^{-j 2π k / N} = cos(2π k/N) - j sin(2π k/N)
 *
 * Tables are stored in split format (separate real/imag arrays) to enable
 * efficient vector loads on RVV hardware.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "rvvfft_types.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* ── Single precision ──────────────────────────────────────────────────────── */

/**
 * rvvfft_twiddle_init_f32 — Fill split twiddle factor tables (f32).
 *
 * @tw_r:   Output real  parts (count entries).
 * @tw_i:   Output imag  parts (count entries).
 * @n:      Full transform length N.
 * @stride: Index stride between twiddle entries (= N / radix for each stage).
 * @count:  Number of entries to fill (= N / radix * (radix-1)).
 */
void rvvfft_twiddle_init_f32(float *tw_r, float *tw_i,
                              int n, int stride, int count)
{
    double two_pi_over_n = -2.0 * M_PI / (double)n;
    for (int k = 0; k < count; ++k) {
        double angle = two_pi_over_n * (double)(k * stride);
        tw_r[k] = (float)cos(angle);
        tw_i[k] = (float)sin(angle);
    }
}

/**
 * rvvfft_twiddle_alloc_f32 — Allocate and initialise a twiddle table (f32).
 *
 * Returns a heap-allocated pair of arrays {tw_r, tw_i} each of length
 * (n / 2), covering all twiddle factors for a length-n FFT.
 * Caller must free both pointers with free().
 *
 * Returns RVVFFT_SUCCESS or RVVFFT_ERR_NO_MEMORY.
 */
rvvfft_status_t rvvfft_twiddle_alloc_f32(int n,
                                          float **tw_r_out,
                                          float **tw_i_out)
{
    if (!rvvfft_is_power_of_2(n) || n < 2)
        return RVVFFT_ERR_INVALID_SIZE;

    int count = n / 2;
    float *tw_r = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                          (size_t)count * sizeof(float));
    float *tw_i = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                          (size_t)count * sizeof(float));
    if (!tw_r || !tw_i) {
        free(tw_r);
        free(tw_i);
        return RVVFFT_ERR_NO_MEMORY;
    }

    rvvfft_twiddle_init_f32(tw_r, tw_i, n, 1, count);
    *tw_r_out = tw_r;
    *tw_i_out = tw_i;
    return RVVFFT_SUCCESS;
}

/* ── Double precision ──────────────────────────────────────────────────────── */

/**
 * rvvfft_twiddle_init_f64 — Fill split twiddle factor tables (f64).
 */
void rvvfft_twiddle_init_f64(double *tw_r, double *tw_i,
                               int n, int stride, int count)
{
    double two_pi_over_n = -2.0 * M_PI / (double)n;
    for (int k = 0; k < count; ++k) {
        double angle = two_pi_over_n * (double)(k * stride);
        tw_r[k] = cos(angle);
        tw_i[k] = sin(angle);
    }
}

/**
 * rvvfft_twiddle_alloc_f64 — Allocate and initialise a twiddle table (f64).
 */
rvvfft_status_t rvvfft_twiddle_alloc_f64(int n,
                                           double **tw_r_out,
                                           double **tw_i_out)
{
    if (!rvvfft_is_power_of_2(n) || n < 2)
        return RVVFFT_ERR_INVALID_SIZE;

    int count = n / 2;
    double *tw_r = (double *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                            (size_t)count * sizeof(double));
    double *tw_i = (double *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                            (size_t)count * sizeof(double));
    if (!tw_r || !tw_i) {
        free(tw_r);
        free(tw_i);
        return RVVFFT_ERR_NO_MEMORY;
    }

    rvvfft_twiddle_init_f64(tw_r, tw_i, n, 1, count);
    *tw_r_out = tw_r;
    *tw_i_out = tw_i;
    return RVVFFT_SUCCESS;
}
