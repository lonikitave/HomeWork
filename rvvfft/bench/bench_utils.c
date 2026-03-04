/**
 * bench_utils.c — Benchmarking utility functions.
 *
 * Provides helpers for:
 *   - GFLOPS calculation
 *   - RISC-V rdcycle / rdtime reading
 *   - Memory bandwidth estimation
 *   - Roofline model computation
 */

#include <math.h>
#include <stdio.h>
#include "rvvfft_types.h"

/* ─── GFLOPS ─────────────────────────────────────────────────────────────── */

/**
 * bench_gflops — Compute measured GFLOPS for an N-point FFT.
 *
 * @n:          Transform length.
 * @reps:       Number of repetitions.
 * @elapsed_s:  Total elapsed time in seconds.
 */
double bench_gflops(int n, int reps, double elapsed_s)
{
    if (elapsed_s <= 0.0) return 0.0;
    /* Standard formula: 5 * N * log2(N) floating-point operations */
    double flops = 5.0 * (double)n * log2((double)n);
    return flops * (double)reps / elapsed_s / 1e9;
}

/**
 * bench_peak_gflops — Compute theoretical peak GFLOPS for an RVV core.
 *
 * @vlen:       Vector length in bits (256 or 1024).
 * @sew:        Scalar element width in bits (32 for f32, 64 for f64).
 * @freq_ghz:   Core frequency in GHz.
 * @fma_per_cy: FMA operations per cycle per vector unit (typically 1 or 2).
 */
double bench_peak_gflops(int vlen, int sew, double freq_ghz, int fma_per_cy)
{
    /* Elements per vector × 2 FLOPS per FMA × FMAs per cycle × freq */
    double elems_per_vec = (double)vlen / (double)sew;
    return elems_per_vec * 2.0 * (double)fma_per_cy * freq_ghz;
}

/**
 * bench_efficiency_pct — Peak efficiency percentage.
 *
 * @measured_gflops: Measured GFLOPS.
 * @peak_gflops:     Theoretical peak GFLOPS.
 */
double bench_efficiency_pct(double measured_gflops, double peak_gflops)
{
    if (peak_gflops <= 0.0) return 0.0;
    return measured_gflops / peak_gflops * 100.0;
}

/* ─── Roofline ───────────────────────────────────────────────────────────── */

/**
 * bench_arithmetic_intensity — Compute arithmetic intensity (FLOP/byte) for FFT.
 *
 * Simplified model: the FFT reads and writes ~2 complex buffers of n elements.
 * Actual cache behaviour is more complex — this gives the memory-bound estimate.
 *
 * @n:    Transform length.
 * @sew:  Scalar element width in bits.
 */
double bench_arithmetic_intensity(int n, int sew)
{
    double flops   = 5.0 * (double)n * log2((double)n);
    /* Two reads + one write = 3 * n complex samples = 6 * n scalars */
    double bytes   = 6.0 * (double)n * (double)(sew / 8);
    return flops / bytes;
}

/**
 * bench_roofline_gflops — Predicted GFLOPS from roofline model.
 *
 * Returns min(peak_compute, peak_bw * arithmetic_intensity).
 *
 * @peak_compute_gflops: Compute roof (GFLOPS).
 * @peak_bw_gbps:        Memory bandwidth roof (GB/s).
 * @intensity:           Arithmetic intensity (FLOP/byte).
 */
double bench_roofline_gflops(double peak_compute_gflops,
                               double peak_bw_gbps,
                               double intensity)
{
    double bw_bound = peak_bw_gbps * intensity;
    return (bw_bound < peak_compute_gflops) ? bw_bound : peak_compute_gflops;
}

/* ─── Reporting ──────────────────────────────────────────────────────────── */

/**
 * bench_print_roofline — Print a summary roofline analysis for a set of sizes.
 *
 * @sizes:   Array of transform lengths.
 * @gflops:  Array of measured GFLOPS (same length as sizes).
 * @count:   Number of entries.
 * @vlen:    Hardware VLEN.
 * @sew:     SEW.
 * @freq_ghz, @fma_per_cy, @peak_bw_gbps: Hardware parameters.
 */
void bench_print_roofline(const int *sizes, const double *gflops, int count,
                           int vlen, int sew,
                           double freq_ghz, int fma_per_cy,
                           double peak_bw_gbps)
{
    double peak_c = bench_peak_gflops(vlen, sew, freq_ghz, fma_per_cy);

    printf("\n── Roofline Analysis (VLEN=%d, SEW=%d, %.2f GHz) ──────────────\n",
           vlen, sew, freq_ghz);
    printf("  Peak compute : %.2f GFLOPS\n", peak_c);
    printf("  Peak BW      : %.2f GB/s\n\n", peak_bw_gbps);
    printf("  %-8s  %-10s  %-10s  %-8s  %-8s  %s\n",
           "n", "AI (F/B)", "Roof (GF)", "Meas (GF)", "Effic (%)", "Bound");
    printf("  %s\n",
           "---------------------------------------------------------------");

    for (int i = 0; i < count; ++i) {
        int    n   = sizes[i];
        double ai  = bench_arithmetic_intensity(n, sew);
        double rf  = bench_roofline_gflops(peak_c, peak_bw_gbps, ai);
        double eff = bench_efficiency_pct(gflops[i], peak_c);
        const char *bound = (ai < peak_c / peak_bw_gbps) ? "BW" : "Compute";

        printf("  %-8d  %-10.2f  %-10.2f  %-8.2f  %-8.1f  %s\n",
               n, ai, rf, gflops[i], eff, bound);
    }
    printf("\n");
}
