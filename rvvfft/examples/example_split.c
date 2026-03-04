/**
 * example_split.c — 分离布局FFT使用示例
 *
 * 演示使用rvvfft执行1D复数FFT（分离布局，RVV最优路径）：
 *   - 实部和虚部在独立数组中存储
 *   - 直接使用rvvfft_plan_dft_split_1d接口
 *   - 演示in-place和out-of-place两种模式
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/rvvfft.h"
#include "../include/rvvfft_types.h"

int main(void)
{
    const int N = 4096;

    printf("rvvfft Split布局示例 — N=%d\n", N);

    rvvfft_init(RVVFFT_VLEN_AUTO);
    printf("  VLEN = %d位\n", rvvfft_get_vlen());

    /* 分配split布局缓冲区（对齐） */
    float *real_in  = rvvfft_aligned_malloc((size_t)N * sizeof(float));
    float *imag_in  = rvvfft_aligned_malloc((size_t)N * sizeof(float));
    float *real_out = rvvfft_aligned_malloc((size_t)N * sizeof(float));
    float *imag_out = rvvfft_aligned_malloc((size_t)N * sizeof(float));

    if (!real_in || !imag_in || !real_out || !imag_out) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    /* 初始化输入：单位冲击信号 x[0]=1，其余为0 */
    memset(real_in, 0, (size_t)N * sizeof(float));
    memset(imag_in, 0, (size_t)N * sizeof(float));
    real_in[0] = 1.0f;

    /* 创建split布局plan（FORWARD） */
    rvvfft_plan plan_fwd = rvvfft_plan_dft_split_1d(
        N, real_in, imag_in, real_out, imag_out,
        RVVFFT_FORWARD, RVVFFT_ESTIMATE);
    if (!plan_fwd) { fprintf(stderr, "创建正向plan失败\n"); return 1; }

    /* 执行正向FFT：单位冲击的FFT应等于全1（每个bin幅值为1） */
    rvvfft_execute_dft_split(plan_fwd, real_in, imag_in, real_out, imag_out);

    /* 验证：所有bin幅值应为1.0 */
    float max_err = 0.0f;
    for (int k = 0; k < N; ++k) {
        float mag = sqrtf(real_out[k]*real_out[k] + imag_out[k]*imag_out[k]);
        float err = fabsf(mag - 1.0f);
        if (err > max_err) max_err = err;
    }
    printf("  单位冲击FFT：最大幅值误差 = %.2e %s\n",
           (double)max_err, max_err < 1e-5f ? "✓" : "✗");

    /* 演示逆FFT：IDFT(DFT(x)) = x（归一化） */
    float *real_rec = rvvfft_aligned_malloc((size_t)N * sizeof(float));
    float *imag_rec = rvvfft_aligned_malloc((size_t)N * sizeof(float));

    rvvfft_plan plan_inv = rvvfft_plan_dft_split_1d(
        N, real_out, imag_out, real_rec, imag_rec,
        RVVFFT_BACKWARD, RVVFFT_ESTIMATE);
    if (!plan_inv) { fprintf(stderr, "创建逆向plan失败\n"); return 1; }

    rvvfft_execute_dft_split(plan_inv, real_out, imag_out, real_rec, imag_rec);

    /* 归一化：IDFT输出需除以N */
    float inv_n = 1.0f / (float)N;
    for (int k = 0; k < N; ++k) {
        real_rec[k] *= inv_n;
        imag_rec[k] *= inv_n;
    }

    /* 验证：real_rec[0]≈1，其余≈0 */
    float err0 = fabsf(real_rec[0] - 1.0f);
    float max_rest = 0.0f;
    for (int k = 1; k < N; ++k) {
        float err = fabsf(real_rec[k]);
        if (err > max_rest) max_rest = err;
    }
    printf("  IDFT(DFT(δ))[0] 误差: %.2e %s\n", (double)err0, err0 < 1e-5f ? "✓" : "✗");
    printf("  IDFT(DFT(δ))[k≠0] 最大值: %.2e %s\n",
           (double)max_rest, max_rest < 1e-5f ? "✓" : "✗");

    /* 清理 */
    rvvfft_destroy_plan(plan_fwd);
    rvvfft_destroy_plan(plan_inv);
    rvvfft_aligned_free(real_in);  rvvfft_aligned_free(imag_in);
    rvvfft_aligned_free(real_out); rvvfft_aligned_free(imag_out);
    rvvfft_aligned_free(real_rec); rvvfft_aligned_free(imag_rec);
    rvvfft_cleanup();

    printf("  示例完成。\n");
    return 0;
}
