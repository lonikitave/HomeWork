#include "rvvfft.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* Read cycle counter (RISC-V) */
static inline uint64_t rdcycle(void)
{
#ifdef __riscv
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

static double gflops(int n, double seconds)
{
    return 5.0 * n * log2(n) / seconds / 1e9;
}

int main(void)
{
    static const int sizes[] = {
        1024, 4096, 16384, 65536, 262144, 1048576, 4194304
    };
    static const int NRUNS = 20;

    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        rvvfft_complex *buf = malloc(n * sizeof(rvvfft_complex));
        if (!buf) { perror("malloc"); return 1; }

        rvvfft_plan plan = rvvfft_plan_dft_1d(
            n, buf, buf, RVVFFT_FORWARD, RVVFFT_ESTIMATE);

        /* Warm-up */
        rvvfft_execute(plan);

        uint64_t t0 = rdcycle();
        for (int r = 0; r < NRUNS; r++)
            rvvfft_execute(plan);
        uint64_t t1 = rdcycle();

        double elapsed_ns = (double)(t1 - t0);
#ifdef __riscv
        /* cycles → seconds: assume 1 GHz for cycle-based counter */
        double seconds = elapsed_ns / 1e9 / NRUNS;
#else
        double seconds = elapsed_ns / 1e9 / NRUNS;
#endif
        printf("N=%7d  %.3f GFLOPS\n", n, gflops(n, seconds));

        rvvfft_destroy_plan(plan);
        free(buf);
    }
    return 0;
}
