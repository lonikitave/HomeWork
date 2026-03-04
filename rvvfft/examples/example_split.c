/**
 * example_split.c — Complex FFT using split (separate real/imag) layout.
 *
 * The split layout stores real and imaginary parts in separate arrays:
 *   real[]: [r0, r1, r2, ...]
 *   imag[]: [i0, i1, i2, ...]
 *
 * This layout is more efficient on RVV hardware because vector loads are
 * always unit-stride (no interleaving overhead).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rvvfft.h"

#define N 1024

int main(void)
{
    rvvfft_init();
    printf("rvvfft split-layout example\n");
    printf("VLEN = %d\n\n", rvvfft_detected_vlen());

    /* ── Allocate split buffers ──────────────────────────────────────── */
    float *in_r  = (float *)rvvfft_malloc(N * sizeof(float));
    float *in_i  = (float *)rvvfft_malloc(N * sizeof(float));
    float *out_r = (float *)rvvfft_malloc(N * sizeof(float));
    float *out_i = (float *)rvvfft_malloc(N * sizeof(float));

    if (!in_r || !in_i || !out_r || !out_i) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* ── Input: real-valued cosine signal ─────────────────────────────── */
    int freq = 16;  /* cycles per N samples */
    for (int k = 0; k < N; ++k) {
        in_r[k] = cosf(2.0f * 3.14159265f * (float)freq * (float)k / (float)N);
        in_i[k] = 0.0f;
    }

    /* ── Plan and execute ────────────────────────────────────────────── */
    rvvfft_plan plan = rvvfft_plan_dft_split_1d(N,
                                                 in_r, in_i,
                                                 out_r, out_i,
                                                 RVVFFT_FORWARD,
                                                 RVVFFT_ESTIMATE);
    if (!plan) { fprintf(stderr, "Planning failed\n"); return 1; }
    rvvfft_execute(plan);

    /* ── Inspect: for a cosine at bin `freq`, expect peaks at ±freq ─── */
    printf("Magnitude spectrum around bin %d (expect peak ≈ N/2 = %d):\n", freq, N/2);
    for (int k = freq - 2; k <= freq + 2; ++k) {
        float mag = sqrtf(out_r[k] * out_r[k] + out_i[k] * out_i[k]);
        printf("  |X[%3d]| = %.2f\n", k, (double)mag);
    }

    rvvfft_destroy_plan(plan);
    rvvfft_free(in_r);
    rvvfft_free(in_i);
    rvvfft_free(out_r);
    rvvfft_free(out_i);
    rvvfft_cleanup();
    return 0;
}
