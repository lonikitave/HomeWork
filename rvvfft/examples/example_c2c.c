/**
 * example_c2c.c — Complex-to-complex FFT example (interleaved layout, f32).
 *
 * Demonstrates:
 *   1. Library initialisation
 *   2. Buffer allocation (aligned)
 *   3. Plan creation (ESTIMATE mode)
 *   4. Forward FFT execution
 *   5. Inverse FFT and verification (round-trip test)
 *   6. Resource cleanup
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rvvfft.h"

#define N 1024   /* Transform length — must be a power of 2 */

int main(void)
{
    rvvfft_init();
    printf("rvvfft version : %s\n", rvvfft_version());
    printf("Detected VLEN  : %d\n\n", rvvfft_detected_vlen());

    /* ── Allocate aligned buffers ──────────────────────────────────── */
    rvvfft_complex *in      = (rvvfft_complex *)rvvfft_malloc(N * sizeof(rvvfft_complex));
    rvvfft_complex *out_fwd = (rvvfft_complex *)rvvfft_malloc(N * sizeof(rvvfft_complex));
    rvvfft_complex *out_inv = (rvvfft_complex *)rvvfft_malloc(N * sizeof(rvvfft_complex));

    if (!in || !out_fwd || !out_inv) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* ── Initialise input: Dirac delta δ[0] → constant spectrum ─────── */
    for (int k = 0; k < N; ++k) {
        in[k].r = (k == 0) ? 1.0f : 0.0f;
        in[k].i = 0.0f;
    }

    /* ── Forward FFT ─────────────────────────────────────────────────── */
    rvvfft_plan plan_fwd = rvvfft_plan_dft_1d(N, in, out_fwd,
                                               RVVFFT_FORWARD, RVVFFT_ESTIMATE);
    if (!plan_fwd) { fprintf(stderr, "Planning failed\n"); return 1; }
    rvvfft_execute(plan_fwd);

    /* δ[0] → all-ones spectrum */
    printf("Forward FFT of δ[0]  — first 4 output bins (expect ~1+0j each):\n");
    for (int k = 0; k < 4; ++k)
        printf("  out[%d] = %+.6f + %+.6fj\n", k,
               (double)out_fwd[k].r, (double)out_fwd[k].i);

    /* ── Inverse FFT ──────────────────────────────────────────────────── */
    rvvfft_plan plan_inv = rvvfft_plan_dft_1d(N, out_fwd, out_inv,
                                               RVVFFT_BACKWARD, RVVFFT_ESTIMATE);
    if (!plan_inv) { fprintf(stderr, "Planning failed\n"); return 1; }
    rvvfft_execute(plan_inv);

    /* Verify round-trip: IFFT(FFT(x)) == x */
    float max_err = 0.0f;
    for (int k = 0; k < N; ++k) {
        float err_r = fabsf(out_inv[k].r - in[k].r);
        float err_i = fabsf(out_inv[k].i - in[k].i);
        if (err_r > max_err) max_err = err_r;
        if (err_i > max_err) max_err = err_i;
    }
    printf("\nRound-trip max absolute error: %.2e  %s\n",
           (double)max_err,
           (max_err < 1e-5f) ? "(PASS)" : "(FAIL)");

    /* ── Cleanup ──────────────────────────────────────────────────────── */
    rvvfft_destroy_plan(plan_fwd);
    rvvfft_destroy_plan(plan_inv);
    rvvfft_free(in);
    rvvfft_free(out_fwd);
    rvvfft_free(out_inv);
    rvvfft_cleanup();
    return 0;
}
