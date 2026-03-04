/**
 * bench_main.c — 性能benchmark主程序
 *
 * 测试rvvfft在各种配置下的性能：
 *   - FFT长度：2^10 ~ 2^24
 *   - 布局：split, interleaved
 *   - 方向：FORWARD
 *   - VLEN：由编译时RVVFFT_VLEN决定
 *
 * 输出格式（CSV和表格两种模式）：
 *   N, VLEN, time_us, GFLOPS, peak%, scalar_speedup
 *
 * 使用方法：
 *   ./bench [--min_log2 10] [--max_log2 20] [--repeats 100]
 *           [--freq 1.0] [--csv]
 *
 * 性能对标方法论参考Phase 4：
 *   峰值效率(%) = Measured_GFLOPS / Theoretical_Peak × 100
 *   目标：≥70%峰值效率，VLEN=1024加速比>10×
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/rvvfft.h"
#include "../include/rvvfft_types.h"

/* 声明 bench_utils.c 中的函数 */
extern uint64_t rvvfft_bench_rdcycle(void);
extern double   rvvfft_bench_time_seconds(void);
extern double   rvvfft_bench_flops(int n);
extern double   rvvfft_bench_gflops(int n, double time_sec, int repeats);
extern double   rvvfft_bench_peak_gflops(int vlen, double freq_ghz,
                                          double fma_per_cycle, int num_cores);
extern void     rvvfft_bench_print_header(void);
extern void     rvvfft_bench_print_result(int n, int vlen, double time_us,
                                           double gflops, double peak_gflops,
                                           double scalar_gflops, int repeats);

/* ── 标量C基线FFT（用于加速比计算） ─────────────────────────── */
extern void rvvfft_fft_iterative_split_f32(float *real, float *imag,
                                            int n, int sign);

static double bench_scalar(int n, int repeats)
{
    float *real = calloc((size_t)n, sizeof(float));
    float *imag = calloc((size_t)n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return 0.0; }

    /* 热身 */
    rvvfft_fft_iterative_split_f32(real, imag, n, RVVFFT_FORWARD);

    double t0 = rvvfft_bench_time_seconds();
    for (int r = 0; r < repeats; ++r) {
        real[0] = (float)r;  /* 防止编译器优化掉 */
        rvvfft_fft_iterative_split_f32(real, imag, n, RVVFFT_FORWARD);
    }
    double elapsed = rvvfft_bench_time_seconds() - t0;

    free(real); free(imag);
    return rvvfft_bench_gflops(n, elapsed, repeats);
}

/* ── rvvfft API benchmark ─────────────────────────────────── */
static double bench_rvvfft_split(int n, int repeats)
{
    float *real_buf = rvvfft_aligned_malloc((size_t)n * 4 * sizeof(float));
    if (!real_buf) return 0.0;
    float *imag_buf = real_buf + n;
    float *ro = real_buf + 2 * n;
    float *io = real_buf + 3 * n;

    /* 初始化数据 */
    for (int i = 0; i < n; ++i) { real_buf[i] = (float)i / n; imag_buf[i] = 0.0f; }

    rvvfft_plan plan = rvvfft_plan_dft_split_1d(
        n, real_buf, imag_buf, ro, io, RVVFFT_FORWARD, RVVFFT_ESTIMATE);
    if (!plan) { rvvfft_aligned_free(real_buf); return 0.0; }

    /* 热身 */
    rvvfft_execute_dft_split(plan, real_buf, imag_buf, ro, io);

    double t0 = rvvfft_bench_time_seconds();
    for (int r = 0; r < repeats; ++r) {
        real_buf[0] = (float)r;
        rvvfft_execute_dft_split(plan, real_buf, imag_buf, ro, io);
    }
    double elapsed = rvvfft_bench_time_seconds() - t0;

    rvvfft_destroy_plan(plan);
    rvvfft_aligned_free(real_buf);
    return rvvfft_bench_gflops(n, elapsed, repeats);
}

/* ── 主函数 ──────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    int    min_log2 = 10;
    int    max_log2 = 20;
    int    repeats  = 50;
    double freq_ghz = 1.0;   /* 芯片频率，需根据实际硬件填写 */
    int    csv_mode = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--min_log2") && i+1 < argc)  min_log2 = atoi(argv[++i]);
        if (!strcmp(argv[i], "--max_log2") && i+1 < argc)  max_log2 = atoi(argv[++i]);
        if (!strcmp(argv[i], "--repeats")  && i+1 < argc)  repeats  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--freq")     && i+1 < argc)  freq_ghz = atof(argv[++i]);
        if (!strcmp(argv[i], "--csv"))  csv_mode = 1;
    }

    rvvfft_init(RVVFFT_VLEN_AUTO);
    int vlen = rvvfft_get_vlen();

    /* 理论峰值：VLEN/32个f32元素 × 2 FLOP/FMA × freq_ghz GHz */
    double peak = rvvfft_bench_peak_gflops(vlen, freq_ghz, 1.0, 1);

    printf("rvvfft 性能Benchmark\n");
    printf("VLEN=%d, 理论峰值=%.3f GFLOPS (freq=%.2fGHz, 1核)\n\n",
           vlen, peak, freq_ghz);

    if (csv_mode) {
        printf("N,VLEN,time_us,rvvfft_GFLOPS,scalar_GFLOPS,peak_pct,speedup\n");
    } else {
        rvvfft_bench_print_header();
    }

    for (int k = min_log2; k <= max_log2; ++k) {
        int n = 1 << k;
        /* 大N减少重复次数避免测试时间过长 */
        int reps = (n >= (1 << 18)) ? 5 : (n >= (1 << 15)) ? 20 : repeats;

        double scalar_gflops = bench_scalar(n, reps > 10 ? 10 : reps);
        double rvv_gflops    = bench_rvvfft_split(n, reps);
        double time_us       = (rvv_gflops > 0) ?
            (rvvfft_bench_flops(n) / (rvv_gflops * 1e9) * 1e6) : 0.0;
        double peak_pct      = (peak > 0) ? rvv_gflops / peak * 100.0 : 0.0;
        double speedup       = (scalar_gflops > 0) ? rvv_gflops / scalar_gflops : 0.0;

        if (csv_mode) {
            printf("%d,%d,%.2f,%.3f,%.3f,%.1f,%.2f\n",
                   n, vlen, time_us, rvv_gflops, scalar_gflops, peak_pct, speedup);
        } else {
            rvvfft_bench_print_result(n, vlen, time_us, rvv_gflops,
                                     peak, scalar_gflops, reps);
        }
        fflush(stdout);
    }

    rvvfft_cleanup();
    return 0;
}
