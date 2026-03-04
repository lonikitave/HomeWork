/**
 * example_c2c.c — C2C复数FFT使用示例（交错布局）
 *
 * 演示使用rvvfft执行1D复数FFT的完整流程：
 *   1. 分配并初始化输入缓冲区（交错布局）
 *   2. 创建ESTIMATE模式plan
 *   3. 执行FFT
 *   4. 验证输出（检查能量守恒 Parseval定理）
 *   5. 销毁plan，释放资源
 *
 * 编译（宿主机x86开发验证）：
 *   gcc -O2 -I../include -o example_c2c example_c2c.c \
 *       ../src/executor/executor_recursive.c \
 *       ../src/executor/executor_iterative.c \
 *       ../src/planner/planner.c ../src/planner/wisdom.c \
 *       ../src/utils/twiddle.c ../src/utils/bitrev.c \
 *       ../src/utils/alignment.c \
 *       ../src/layout/interleaved.c ../src/layout/split.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/rvvfft.h"
#include "../include/rvvfft_types.h"

int main(void)
{
    const int N = 1024;

    printf("rvvfft C2C示例 — N=%d, 交错布局\n", N);

    /* 1. 初始化库 */
    rvvfft_init(RVVFFT_VLEN_AUTO);
    printf("  VLEN = %d位\n", rvvfft_get_vlen());

    /* 2. 分配对齐缓冲区 */
    rvvfft_complex *in  = rvvfft_aligned_malloc((size_t)N * sizeof(rvvfft_complex));
    rvvfft_complex *out = rvvfft_aligned_malloc((size_t)N * sizeof(rvvfft_complex));
    if (!in || !out) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    /* 3. 初始化输入：单频信号 cos(2π·k·4/N) */
    for (int k = 0; k < N; ++k) {
        double theta = 2.0 * 3.14159265358979323846 * 4 * k / N;
        in[k].r = (float)cos(theta);
        in[k].i = 0.0f;
    }

    /* 4. 创建ESTIMATE plan（快速，无实际计时） */
    rvvfft_plan plan = rvvfft_plan_dft_1d(N, in, out, RVVFFT_FORWARD, RVVFFT_ESTIMATE);
    if (!plan) {
        fprintf(stderr, "Plan创建失败\n");
        rvvfft_aligned_free(in);
        rvvfft_aligned_free(out);
        return 1;
    }

    /* 5. 执行FFT */
    rvvfft_execute_dft(plan, in, out);
    printf("  FFT执行完成\n");

    /* 6. 验证：Parseval能量守恒定理
     *    Σ|x[k]|² = (1/N) × Σ|X[k]|²  */
    double energy_in = 0.0, energy_out = 0.0;
    for (int k = 0; k < N; ++k) {
        energy_in  += (double)in[k].r * in[k].r + (double)in[k].i * in[k].i;
    }
    for (int k = 0; k < N; ++k) {
        energy_out += (double)out[k].r * out[k].r + (double)out[k].i * out[k].i;
    }
    energy_out /= N;

    double rel_err = fabs(energy_in - energy_out) / (energy_in + 1e-30);
    printf("  输入能量:  %.6f\n", energy_in);
    printf("  输出能量/N:%.6f\n", energy_out);
    printf("  Parseval误差: %.2e %s\n", rel_err,
           rel_err < 1e-4 ? "✓" : "✗");

    /* 7. 检查频谱峰值位于正确位置（bin 4和N-4） */
    float max_mag = 0.0f;
    int   max_bin = 0;
    for (int k = 0; k < N; ++k) {
        float mag = sqrtf(out[k].r * out[k].r + out[k].i * out[k].i);
        if (mag > max_mag) { max_mag = mag; max_bin = k; }
    }
    printf("  频谱峰值位于 bin %d（期望 4），幅值 %.2f\n",
           max_bin, (double)max_mag);
    printf("  结果: %s\n", (max_bin == 4 || max_bin == N - 4) ? "✓ 通过" : "✗ 失败");

    /* 8. 清理 */
    rvvfft_destroy_plan(plan);
    rvvfft_aligned_free(in);
    rvvfft_aligned_free(out);
    rvvfft_cleanup();

    return 0;
}
