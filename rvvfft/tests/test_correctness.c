/*
 * rvvfft correctness test driver.
 *
 * Reads golden binary files produced by tests/gen_golden.py and checks that
 * rvvfft_execute produces results within the allowed ULP tolerance.
 *
 * Usage:  ./test_correctness  tests/golden/
 */

#include "rvvfft.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#define MAX_ULP_F32  4
#define MAX_REL_ERR  1e-6f

static int ulp_f32(float a, float b)
{
    if (a == b) return 0;
    union { float f; int i; } ua = {a}, ub = {b};
    return abs(ua.i - ub.i);
}

static int test_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }

    int n;
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return 1; }

    float *xre = malloc(n * sizeof(float));
    float *xim = malloc(n * sizeof(float));
    float *yre = malloc(n * sizeof(float));
    float *yim = malloc(n * sizeof(float));
    if (!xre || !xim || !yre || !yim) { fclose(f); return 1; }

    fread(xre, sizeof(float), n, f);
    fread(xim, sizeof(float), n, f);
    fread(yre, sizeof(float), n, f);
    fread(yim, sizeof(float), n, f);
    fclose(f);

    /* Build rvvfft input */
    rvvfft_complex *in  = malloc(n * sizeof(rvvfft_complex));
    rvvfft_complex *out = malloc(n * sizeof(rvvfft_complex));
    for (int i = 0; i < n; i++) { in[i].re = xre[i]; in[i].im = xim[i]; }

    rvvfft_plan plan = rvvfft_plan_dft_1d(
        n, in, out, RVVFFT_FORWARD, RVVFFT_ESTIMATE);
    rvvfft_execute(plan);
    rvvfft_destroy_plan(plan);

    int max_ulp = 0, failures = 0;
    for (int i = 0; i < n; i++) {
        int ulp_re = ulp_f32(out[i].re, yre[i]);
        int ulp_im = ulp_f32(out[i].im, yim[i]);
        int ulp = ulp_re > ulp_im ? ulp_re : ulp_im;
        if (ulp > max_ulp) max_ulp = ulp;
        if (ulp > MAX_ULP_F32) failures++;
    }

    printf("  %-40s  N=%6d  max_ulp=%d  %s\n",
           path, n, max_ulp, failures ? "FAIL" : "PASS");

    free(xre); free(xim); free(yre); free(yim);
    free(in);  free(out);
    return failures > 0 ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *golden_dir = argc > 1 ? argv[1] : "tests/golden";
    DIR *d = opendir(golden_dir);
    if (!d) { perror(golden_dir); return 1; }

    int total = 0, failed = 0;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (!strstr(ent->d_name, ".bin")) continue;

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", golden_dir, ent->d_name);
        failed += test_file(path);
        total++;
    }
    closedir(d);

    printf("\n%d/%d tests passed.\n", total - failed, total);
    return failed > 0 ? 1 : 0;
}
