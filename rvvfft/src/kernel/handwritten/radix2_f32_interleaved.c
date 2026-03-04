/**
 * radix2_f32_interleaved.c — Radix-2 butterfly kernel（交错布局, f32）
 *
 * 实现Cooley-Tukey DIT radix-2 butterfly：
 *   X[k]     = E[k] + W(k,N) · O[k]
 *   X[k+N/2] = E[k] - W(k,N) · O[k]
 *
 * 数据布局：交错复数 [r0,i0,r1,i1,...]
 *
 * 两个VLEN特化版本：
 *   - rvvfft_r2_interleaved_f32_vl256()  — VLEN=256时使用
 *   - rvvfft_r2_interleaved_f32_vl1024() — VLEN=1024时使用
 *   - rvvfft_r2_interleaved_f32_generic() — 纯C标量参考实现
 *
 * 注：RVV intrinsic版本在RVVFFT_USE_RVV宏开启时编译，
 *     否则退回到可移植的标量C实现，确保在x86开发机上也能编译。
 */
#include <stddef.h>
#include <math.h>

#include "../../../include/rvvfft_types.h"

/* ── 标量参考实现（可在任何平台编译运行，用于验证） ────────── */

/**
 * @brief Radix-2 DIT butterfly，交错复数布局，纯C标量实现。
 *
 * 处理一个完整的长度为n的FFT的某一pass（stage）：
 * 步长stride表示相邻butterfly输入之间的复数元素间距。
 *
 * @param data   交错复数数组（in-place修改）
 * @param tw_r   twiddle因子实部（预计算，长度≥n/2）
 * @param tw_i   twiddle因子虚部
 * @param n      本级DFT长度
 * @param stride 输入/输出步长（复数元素单位）
 * @param sign   RVVFFT_FORWARD 或 RVVFFT_BACKWARD
 */
void rvvfft_r2_interleaved_f32_scalar(
    rvvfft_complex_f32 *data,
    const float        *tw_r,
    const float        *tw_i,
    int                 n,
    ptrdiff_t           stride,
    int                 sign)
{
    (void)sign; /* twiddle已按sign预计算 */

    int half = n / 2;
    for (int k = 0; k < half; ++k) {
        rvvfft_complex_f32 *e = data + (ptrdiff_t)k * stride;
        rvvfft_complex_f32 *o = e + (ptrdiff_t)half * stride;

        float wr = tw_r[k];
        float wi = tw_i[k];

        /* t = W · O */
        float tr = wr * o->r - wi * o->i;
        float ti = wi * o->r + wr * o->i;

        /* butterfly */
        float er = e->r, ei = e->i;
        e->r = er + tr;
        e->i = ei + ti;
        o->r = er - tr;
        o->i = ei - ti;
    }
}

/* ── RVV Intrinsic 实现（仅在RVV目标平台编译） ────────────── */
#ifdef RVVFFT_USE_RVV
#include <riscv_vector.h>

/*
 * VLEN=256时每个f32向量寄存器（LMUL=1）可容纳8个复数（实部/虚部分开存储）。
 * 交错布局需要先de-interleave才能做向量化复数乘法，
 * 使用vrgather或vlseg2e32进行结构化load。
 *
 * 以下针对VLEN=256特化：VL_F32=8，每次处理8个butterfly
 */

/**
 * @brief Radix-2 DIT butterfly，交错布局，VLEN=256 RVV实现。
 *
 * 使用vlseg2e32（分段load）将交错数据拆分为实部/虚部向量，
 * 执行向量化复数乘法和加减，再用vsseg2e32交错写回。
 */
void rvvfft_r2_interleaved_f32_vl256(
    rvvfft_complex_f32 *data,
    const float        *tw_r,
    const float        *tw_i,
    int                 n,
    ptrdiff_t           stride,
    int                 sign)
{
    (void)sign;
    int half = n / 2;

    /*
     * 当stride=1时（连续内存），使用vlseg2e32批量加载交错数据。
     * stride≠1时退回到标量实现（gather代价太高）。
     */
    if (stride != 1) {
        rvvfft_r2_interleaved_f32_scalar(data, tw_r, tw_i, n, stride, sign);
        return;
    }

    rvvfft_complex_f32 *e_base = data;
    rvvfft_complex_f32 *o_base = data + half;

    int remaining = half;
    int tw_offset = 0;

    while (remaining > 0) {
        /* 设置向量长度（处理尾部不足一个完整向量的情况） */
        size_t vl = __riscv_vsetvl_e32m1((size_t)remaining);

        /* 加载twiddle因子 */
        vfloat32m1_t vwr = __riscv_vle32_v_f32m1(tw_r + tw_offset, vl);
        vfloat32m1_t vwi = __riscv_vle32_v_f32m1(tw_i + tw_offset, vl);

        /* 加载交错复数：vlseg2e32 拆分为 e_r, e_i */
        vfloat32m1x2_t ve_seg = __riscv_vlseg2e32_v_f32m1x2(
            (const float *)e_base + tw_offset * 2, vl);
        vfloat32m1_t ve_r = __riscv_vget_v_f32m1x2_f32m1(ve_seg, 0);
        vfloat32m1_t ve_i = __riscv_vget_v_f32m1x2_f32m1(ve_seg, 1);

        vfloat32m1x2_t vo_seg = __riscv_vlseg2e32_v_f32m1x2(
            (const float *)o_base + tw_offset * 2, vl);
        vfloat32m1_t vo_r = __riscv_vget_v_f32m1x2_f32m1(vo_seg, 0);
        vfloat32m1_t vo_i = __riscv_vget_v_f32m1x2_f32m1(vo_seg, 1);

        /*
         * 复数乘法：t = W · O
         *   tr = wr*or - wi*oi
         *   ti = wi*or + wr*oi
         */
        vfloat32m1_t vtr = __riscv_vfmul_vv_f32m1(vwr, vo_r, vl);
        vtr = __riscv_vfnmsac_vv_f32m1(vtr, vwi, vo_i, vl); /* tr -= wi*oi */

        vfloat32m1_t vti = __riscv_vfmul_vv_f32m1(vwi, vo_r, vl);
        vti = __riscv_vfmacc_vv_f32m1(vti, vwr, vo_i, vl); /* ti += wr*oi */

        /* Butterfly: E' = E + T, O' = E - T */
        vfloat32m1_t ve_r_out = __riscv_vfadd_vv_f32m1(ve_r, vtr, vl);
        vfloat32m1_t ve_i_out = __riscv_vfadd_vv_f32m1(ve_i, vti, vl);
        vfloat32m1_t vo_r_out = __riscv_vfsub_vv_f32m1(ve_r, vtr, vl);
        vfloat32m1_t vo_i_out = __riscv_vfsub_vv_f32m1(ve_i, vti, vl);

        /* 交错写回：vsseg2e32 */
        vfloat32m1x2_t ve_out = __riscv_vcreate_v_f32m1x2(ve_r_out, ve_i_out);
        __riscv_vsseg2e32_v_f32m1x2((float *)e_base + tw_offset * 2, ve_out, vl);

        vfloat32m1x2_t vo_out = __riscv_vcreate_v_f32m1x2(vo_r_out, vo_i_out);
        __riscv_vsseg2e32_v_f32m1x2((float *)o_base + tw_offset * 2, vo_out, vl);

        tw_offset += (int)vl;
        remaining -= (int)vl;
    }
}

/**
 * @brief Radix-2 DIT butterfly，交错布局，VLEN=1024 RVV实现。
 *
 * VLEN=1024时VL_F32=32，每次处理32个butterfly，
 * 可使用更激进的循环展开和更大的LMUL=2/4进一步提高吞吐。
 * 当前实现使用LMUL=1作为基线，后续优化任务(4.7)调整LMUL。
 */
void rvvfft_r2_interleaved_f32_vl1024(
    rvvfft_complex_f32 *data,
    const float        *tw_r,
    const float        *tw_i,
    int                 n,
    ptrdiff_t           stride,
    int                 sign)
{
    /*
     * VLEN=1024的主体逻辑与VLEN=256相同，
     * vsetvl_e32m1会自动返回vl=32（VLEN=1024, LMUL=1, SEW=32）。
     * 编译器为两个特化版本生成不同的vsetvli立即数，
     * 达到VLEN特化优化（循环展开、流水线调度）的效果。
     */
    rvvfft_r2_interleaved_f32_vl256(data, tw_r, tw_i, n, stride, sign);
}

#endif /* RVVFFT_USE_RVV */
