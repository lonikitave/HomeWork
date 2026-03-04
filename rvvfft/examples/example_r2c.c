/**
 * example_r2c.c — 实数FFT (R2C) 使用示例
 *
 * 演示rvvfft的R2C（实数到复数）FFT：
 *   - 利用共轭对称性，只需计算n/2+1个输出频点
 *   - 输入为n个实数，输出为n/2+1个复数
 *   - 比C2C快约2倍（实际为1.5-1.8×，取决于实现）
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/rvvfft.h"
#include "../include/rvvfft_types.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

int main(void)
{
    const int N = 2048;

    printf("rvvfft R2C示例 — N=%d（实数FFT）\n", N);

    rvvfft_init(RVVFFT_VLEN_AUTO);

    /* 分配输入（实数）和输出（复数，n/2+1个频点） */
    float          *real_in = rvvfft_aligned_malloc((size_t)N * sizeof(float));
    rvvfft_complex *cplx_out = rvvfft_aligned_malloc(
        (size_t)(N / 2 + 1) * sizeof(rvvfft_complex));

    if (!real_in || !cplx_out) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    /* 初始化：叠加3个频率分量 (freq=10, 50, 200 Hz, 采样率N Hz) */
    for (int k = 0; k < N; ++k) {
        double t = (double)k / N;
        real_in[k] = (float)(
            1.0 * cos(2 * M_PI * 10  * t) +
            0.5 * cos(2 * M_PI * 50  * t) +
            0.3 * cos(2 * M_PI * 200 * t));
    }

    /* 创建R2C plan */
    rvvfft_plan plan = rvvfft_plan_dft_r2c_1d(N, real_in, cplx_out, RVVFFT_ESTIMATE);
    if (!plan) {
        fprintf(stderr, "R2C plan创建失败\n");
        rvvfft_aligned_free(real_in);
        rvvfft_aligned_free(cplx_out);
        return 1;
    }

    /*
     * 注：当前R2C实现通过length-N/2的C2C FFT近似（Phase 3完善），
     * 此示例演示API调用流程，数值结果为近似值。
     */
    rvvfft_execute(plan);

    /* 显示前10个频点的幅值（期望在bin 10, 50, 200处有峰值） */
    printf("  频域幅值（前%d个频点，采样率=%dHz）:\n", 20, N);
    printf("  %-6s %-8s\n", "freq", "magnitude");
    for (int k = 0; k <= 20 && k <= N / 2; ++k) {
        float mag = sqrtf(cplx_out[k].r * cplx_out[k].r +
                          cplx_out[k].i * cplx_out[k].i) * 2.0f / N;
        printf("  %-6d %-8.4f%s\n", k, (double)mag,
               (k == 10 || k == 50) ? " ← 峰值" : "");
    }

    rvvfft_destroy_plan(plan);
    rvvfft_aligned_free(real_in);
    rvvfft_aligned_free(cplx_out);
    rvvfft_cleanup();

    printf("  R2C示例完成。\n");
    return 0;
}
