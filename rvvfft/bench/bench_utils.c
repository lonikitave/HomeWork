#include <math.h>

/* Compute GFLOPS for a 1-D complex FFT of length n in elapsed seconds. */
double rvvfft_bench_gflops(int n, double seconds)
{
    return 5.0 * n * log2(n) / seconds / 1e9;
}
