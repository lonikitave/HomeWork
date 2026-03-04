/**
 * bench_utils.c — 性能测量工具
 *
 * 提供GFLOPS计算、rdcycle计时、性能结果格式化输出等工具函数。
 * 支持RISC-V rdcycle计时（硬件counter）和POSIX clock_gettime两种模式。
 *
 * GFLOPS计算（标准FFT FLOP定义）：
 *   FFT_FLOPS(N) = 5 × N × log₂(N)
 *   GFLOPS = FFT_FLOPS(N) / time_seconds / 1e9
 *
 * 峰值效率计算（对标Intel MKL的方法论）：
 *   效率(%) = Measured_GFLOPS / Theoretical_Peak_GFLOPS × 100
 *   Theoretical_Peak = (VLEN/SEW) × FREQ_GHz × FMA_OPS × NUM_CORES × 1e9
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "../include/rvvfft_types.h"

/* ── 计时接口 ─────────────────────────────────────────────── */

/**
 * @brief 读取RISC-V cycle计数器（rdcycle）。
 *
 * 在真实RISC-V硬件上使用rdcycle指令；
 * 在x86/模拟器上退回到clock_gettime。
 *
 * @return 当前cycle计数（uint64_t）
 */
uint64_t rvvfft_bench_rdcycle(void)
{
#ifdef __riscv
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/**
 * @brief 获取高精度时间戳（秒，双精度）。
 */
double rvvfft_bench_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── GFLOPS计算 ───────────────────────────────────────────── */

/**
 * @brief 计算FFT的理论FLOP数。
 *
 * 标准计算公式：5 × N × log₂(N)
 *   - N次复数乘法（2次实数乘 + 2次实数加 = 4 FLOP），
 *     但FFT通过对称性减少到5 FLOP/element×stage。
 *
 * @param n    FFT长度
 * @return     总FLOP数（double）
 */
double rvvfft_bench_flops(int n)
{
    return 5.0 * n * log2((double)n);
}

/**
 * @brief 从测量时间计算GFLOPS。
 *
 * @param n          FFT长度
 * @param time_sec   执行时间（秒）
 * @param repeats    重复次数
 * @return           GFLOPS
 */
double rvvfft_bench_gflops(int n, double time_sec, int repeats)
{
    if (time_sec <= 0.0) return 0.0;
    return rvvfft_bench_flops(n) * repeats / time_sec / 1e9;
}

/* ── 性能结果输出 ─────────────────────────────────────────── */

/**
 * @brief 打印性能测试结果表头。
 */
void rvvfft_bench_print_header(void)
{
    printf("%-8s %-8s %-10s %-10s %-10s %-10s %-10s\n",
           "N", "VLEN", "time_us", "GFLOPS", "peak%", "scalar_Xfactor", "repeats");
    printf("%-8s %-8s %-10s %-10s %-10s %-10s %-10s\n",
           "--------", "--------", "----------", "----------",
           "----------", "----------", "----------");
}

/**
 * @brief 打印单条性能测试结果。
 *
 * @param n              FFT长度
 * @param vlen           VLEN（位）
 * @param time_us        平均执行时间（微秒）
 * @param gflops         GFLOPS
 * @param peak_gflops    理论峰值GFLOPS（用于计算效率%）
 * @param scalar_gflops  标量C基线GFLOPS（用于计算加速比）
 * @param repeats        重复次数
 */
void rvvfft_bench_print_result(int n, int vlen, double time_us,
                                double gflops, double peak_gflops,
                                double scalar_gflops, int repeats)
{
    double peak_pct = (peak_gflops > 0) ? gflops / peak_gflops * 100.0 : 0.0;
    double speedup  = (scalar_gflops > 0) ? gflops / scalar_gflops : 0.0;

    printf("%-8d %-8d %-10.2f %-10.3f %-10.1f %-10.2f %-10d\n",
           n, vlen, time_us, gflops, peak_pct, speedup, repeats);
}

/**
 * @brief 计算理论峰值GFLOPS。
 *
 * @param vlen       VLEN（位）
 * @param freq_ghz   芯片频率（GHz）
 * @param fma_per_cycle 每周期FMA次数（通常=1对于单issue向量FPU）
 * @param num_cores  核数
 * @return           理论峰值GFLOPS（f32, LMUL=1）
 */
double rvvfft_bench_peak_gflops(int vlen, double freq_ghz,
                                 double fma_per_cycle, int num_cores)
{
    double vl_f32 = (double)vlen / 32.0;  /* 每vector寄存器f32元素数 */
    /* 每FMA处理：vl个乘法 + vl个加法 = 2×vl FLOP */
    return vl_f32 * 2.0 * fma_per_cycle * freq_ghz * num_cores;
}
