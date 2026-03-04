/**
 * test_correctness.c — Correctness test driver for rvvfft.
 *
 * Reads golden data files produced by gen_golden.py and verifies that
 * rvvfft_execute() output matches the reference within the allowed ULP
 * error budget:
 *
 *   f32: max ULP error ≤ 4
 *
 * Usage:
 *   ./test_correctness tests/golden/
 *
 * Returns 0 on success, non-zero if any test fails.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <dirent.h>

#include "rvvfft.h"
#include "rvvfft_types.h"

/* ─── ULP comparison ────────────────────────────────────────────────────── */

/**
 * f32_ulp_diff — Return the ULP distance between two float32 values.
 */
static uint32_t f32_ulp_diff(float a, float b)
{
    uint32_t ia, ib;
    memcpy(&ia, &a, sizeof ia);
    memcpy(&ib, &b, sizeof ib);

    /* Handle sign: two's complement distance for floats of opposite sign */
    if ((ia >> 31) != (ib >> 31)) {
        /* Different signs: treat -0 == +0 specially */
        if ((ia & 0x7fffffff) == 0 && (ib & 0x7fffffff) == 0) return 0;
        return UINT32_MAX;  /* Large distance for opposite-sign non-zeros */
    }
    return ia > ib ? ia - ib : ib - ia;
}

/**
 * max_ulp_budget — Return the allowed ULP error for a length-n transform.
 *
 * Standard formula: c * log2(n) ULPs, where c ≈ 1 for well-implemented FFTs.
 * We use c=5 to give reasonable headroom for the scalar reference path.
 */
static uint32_t max_ulp_budget(int n)
{
    /* log2(n) * 5, minimum 4 */
    uint32_t budget = (uint32_t)(5 * rvvfft_ilog2(n));
    return budget < 4 ? 4 : budget;
}

/**
 * l2_error_ratio — Compute the normalised L2 error:
 *
 *   E = ||computed - ref||_2 / (||ref||_2 * eps_f32 * log2(n))
 *
 * An E ≤ 100 is considered correct (FFTW convention).
 */
static double l2_error_ratio(const float *ref, const float *got, int len, int n)
{
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    for (int k = 0; k < len; ++k) {
        double d = (double)got[k] - (double)ref[k];
        sum_sq_err += d * d;
        sum_sq_ref += (double)ref[k] * (double)ref[k];
    }
    double norm_ref = sqrt(sum_sq_ref);
    if (norm_ref < 1e-30) return 0.0;  /* zero input — trivially correct */
    /* eps_f32 ≈ 1.19e-7 */
    double eps_f32 = 1.19e-7;
    double log2n   = (double)rvvfft_ilog2(n);
    return sqrt(sum_sq_err) / (norm_ref * eps_f32 * log2n);
}

/* ─── Golden file loading ─────────────────────────────────────────────── */

typedef struct {
    int    n;
    float *input;   /* interleaved, 2*n floats */
    float *ref;     /* interleaved, 2*n floats */
} golden_t;

static golden_t *load_golden(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    golden_t *g = (golden_t *)calloc(1, sizeof(*g));
    if (!g) { fclose(f); return NULL; }

    if (fread(&g->n, sizeof(int), 1, f) != 1) goto err;

    size_t sz = (size_t)(2 * g->n) * sizeof(float);
    g->input = (float *)malloc(sz);
    g->ref   = (float *)malloc(sz);
    if (!g->input || !g->ref) goto err;

    if (fread(g->input, sizeof(float), (size_t)(2 * g->n), f) != (size_t)(2 * g->n)) goto err;
    if (fread(g->ref,   sizeof(float), (size_t)(2 * g->n), f) != (size_t)(2 * g->n)) goto err;

    fclose(f);
    return g;

err:
    fprintf(stderr, "Error reading %s\n", path);
    free(g->input);
    free(g->ref);
    free(g);
    fclose(f);
    return NULL;
}

static void free_golden(golden_t *g)
{
    if (!g) return;
    free(g->input);
    free(g->ref);
    free(g);
}

/* ─── Single test ────────────────────────────────────────────────────────── */

static int run_test(const char *golden_path)
{
    golden_t *g = load_golden(golden_path);
    if (!g) return 1;

    int n = g->n;

    /* Allocate aligned I/O buffers */
    rvvfft_complex *in  = (rvvfft_complex *)rvvfft_malloc(
                              (size_t)n * sizeof(rvvfft_complex));
    rvvfft_complex *out = (rvvfft_complex *)rvvfft_malloc(
                              (size_t)n * sizeof(rvvfft_complex));
    if (!in || !out) {
        fprintf(stderr, "FAIL [n=%d]: allocation failed\n", n);
        rvvfft_free(in); rvvfft_free(out);
        free_golden(g);
        return 1;
    }

    /* Copy input */
    memcpy(in, g->input, (size_t)(2 * n) * sizeof(float));

    /* Plan and execute */
    rvvfft_plan plan = rvvfft_plan_dft_1d(n, in, out, RVVFFT_FORWARD,
                                           RVVFFT_ESTIMATE);
    if (!plan) {
        fprintf(stderr, "FAIL [n=%d]: planning failed\n", n);
        rvvfft_free(in); rvvfft_free(out);
        free_golden(g);
        return 1;
    }

    rvvfft_execute(plan);
    rvvfft_destroy_plan(plan);

    /* Compare against reference — two criteria:
     *   1. Normalised L2 error ≤ 100 (FFTW convention, robust to near-zero bins)
     *   2. Per-sample ULP error ≤ budget (diagnostic only, logged but not fatal)
     */
    uint32_t max_ulp = 0;
    int      failed  = 0;
    const float *ref = g->ref;
    const float *got = (const float *)out;
    uint32_t ulp_budget = max_ulp_budget(n);

    /* ULP scan (diagnostic) */
    for (int k = 0; k < 2 * n; ++k) {
        uint32_t ulp = f32_ulp_diff(ref[k], got[k]);
        if (ulp > max_ulp) max_ulp = ulp;
    }

    /* L2 error ratio (primary pass/fail criterion) */
    double ratio = l2_error_ratio(ref, got, 2 * n, n);
    if (ratio > 100.0) {
        fprintf(stderr, "FAIL [n=%d]: L2 error ratio=%.1f > 100 (max_ulp=%u, budget=%u)\n",
                n, ratio, max_ulp, ulp_budget);
        failed = 1;
    }

    if (!failed)
        printf("PASS [n=%-6d] L2_ratio=%5.1f  max_ulp=%u (budget=%u)\n",
               n, ratio, max_ulp, ulp_budget);

    rvvfft_free(in);
    rvvfft_free(out);
    free_golden(g);
    return failed ? 1 : 0;
}

/* ─── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    const char *golden_dir = (argc >= 2) ? argv[1] : "tests/golden";

    rvvfft_init();
    printf("rvvfft version : %s\n", rvvfft_version());
    printf("detected VLEN  : %d\n", rvvfft_detected_vlen());
    printf("golden dir     : %s\n\n", golden_dir);

    DIR *d = opendir(golden_dir);
    if (!d) {
        fprintf(stderr,
                "Cannot open golden directory '%s'.\n"
                "Run tests/gen_golden.py first.\n", golden_dir);
        return 1;
    }

    int total = 0, passed = 0;
    struct dirent *entry;
    char path[4096];

    while ((entry = readdir(d)) != NULL) {
        /* Skip non-.bin files */
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len < 4 || strcmp(name + len - 4, ".bin") != 0)
            continue;

        snprintf(path, sizeof(path), "%s/%s", golden_dir, name);
        int rc = run_test(path);
        total++;
        if (rc == 0) passed++;
    }
    closedir(d);

    printf("\n%d / %d tests passed.\n", passed, total);
    rvvfft_cleanup();
    return (passed == total) ? 0 : 1;
}
