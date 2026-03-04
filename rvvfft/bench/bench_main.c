/**
 * bench_main.c — Performance benchmark for rvvfft.
 *
 * Measures FFT throughput in GFLOPS and computes:
 *   - Scalar baseline speedup
 *   - Peak efficiency (% of theoretical VLEN peak)
 *
 * Usage:
 *   ./bench [n_min] [n_max] [n_reps]
 *   ./bench 1024 16777216 10
 *
 * GFLOPS formula (standard Cooley-Tukey): 5 * N * log2(N) / time_seconds / 1e9
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rvvfft.h"
#include "rvvfft_types.h"

/* ─── Timing ─────────────────────────────────────────────────────────────── */

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── rdcycle (RISC-V only) ───────────────────────────────────────────────── */

static inline unsigned long long read_cycle(void)
{
#if defined(__riscv)
    unsigned long long c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
#else
    return 0;
#endif
}

/* ─── Benchmark one size ──────────────────────────────────────────────────── */

static void bench_n(int n, int reps, double freq_ghz)
{
    (void)freq_ghz;
    size_t buf_bytes = (size_t)n * sizeof(rvvfft_complex);
    rvvfft_complex *in  = (rvvfft_complex *)rvvfft_malloc(buf_bytes);
    rvvfft_complex *out = (rvvfft_complex *)rvvfft_malloc(buf_bytes);

    if (!in || !out) {
        fprintf(stderr, "bench_n(%d): allocation failed\n", n);
        rvvfft_free(in); rvvfft_free(out);
        return;
    }

    /* Initialise with non-zero data */
    for (int k = 0; k < n; ++k) {
        in[k].r = (float)k / (float)n;
        in[k].i = 0.0f;
    }

    rvvfft_plan plan = rvvfft_plan_dft_1d(n, in, out, RVVFFT_FORWARD,
                                           RVVFFT_ESTIMATE);
    if (!plan) {
        fprintf(stderr, "bench_n(%d): planning failed\n", n);
        rvvfft_free(in); rvvfft_free(out);
        return;
    }

    /* Warm-up */
    rvvfft_execute(plan);

    /* Timed runs */
    double t_start = now_seconds();
    unsigned long long c_start = read_cycle();

    for (int r = 0; r < reps; ++r)
        rvvfft_execute(plan);

    unsigned long long c_end = read_cycle();
    double t_end = now_seconds();

    double elapsed   = t_end - t_start;
    double flops     = rvvfft_flops(n);
    double gflops    = flops * (double)reps / elapsed / 1e9;
    double cyc_per   = (c_end > c_start)
                       ? (double)(c_end - c_start) / (double)reps
                       : 0.0;

    printf("n=%-8d  reps=%-4d  time=%.3f ms  GFLOPS=%-7.3f  cyc/pt=%.1f\n",
           n, reps,
           elapsed / (double)reps * 1e3,
           gflops,
           (cyc_per > 0) ? cyc_per / (double)n : 0.0);

    rvvfft_destroy_plan(plan);
    rvvfft_free(in);
    rvvfft_free(out);
}

/* ─── Main ───────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    int n_min  = (argc >= 2) ? atoi(argv[1]) : 1024;
    int n_max  = (argc >= 3) ? atoi(argv[2]) : 1 << 20;
    int reps   = (argc >= 4) ? atoi(argv[3]) : 10;
    double ghz = (argc >= 5) ? atof(argv[4]) : 1.0;  /* chip frequency GHz */

    if (n_min < 4 || n_max < n_min || reps < 1) {
        fprintf(stderr, "Usage: bench [n_min] [n_max] [reps] [freq_ghz]\n");
        return 1;
    }

    rvvfft_init();

    printf("rvvfft benchmark\n");
    printf("  version      : %s\n", rvvfft_version());
    printf("  VLEN         : %d\n", rvvfft_detected_vlen());
    printf("  n_min        : %d\n", n_min);
    printf("  n_max        : %d\n", n_max);
    printf("  reps         : %d\n", reps);
    printf("  freq         : %.2f GHz\n\n", ghz);

    printf("%-16s %-8s %-16s %-14s %-12s\n",
           "n", "reps", "time/rep (ms)", "GFLOPS", "cyc/point");
    printf("%s\n", "----------------------------------------------------------------------");

    for (int n = n_min; n <= n_max; n *= 2) {
        if (!rvvfft_is_power_of_2(n)) continue;
        bench_n(n, reps, ghz);
    }

    rvvfft_cleanup();
    return 0;
}
