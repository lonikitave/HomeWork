/**
 * radix2_f32_split.c — Radix-2 butterfly kernel（分离布局, f32）
 *
 * 分离(split)布局：实部和虚部分别存储在独立数组中。
 *   real[]: [r0, r1, r2, ..., r_{n-1}]
 *   imag[]: [i0, i1, i2, ..., i_{n-1}]
 *
 * 优势：RVV向量load/store全部为连续访问，无需de-interleave，
 *       天然适合向量化，性能通常优于交错布局约10-20%。
 *
 * 同样提供三个版本：标量参考 / VLEN=256 RVV / VLEN=1024 RVV
 */
#include <stddef.h>
#include <math.h>

#include "../../../include/rvvfft_types.h"

/* ── 标量参考实现 ─────────────────────────────────────────── */

/**
 * @brief Radix-2 DIT butterfly，分离布局，纯C标量实现。
 *
 * @param real   实部数组（in-place修改）
 * @param imag   虚部数组（in-place修改）
 * @param tw_r   twiddle因子实部
 * @param tw_i   twiddle因子虚部
 * @param n      本级DFT长度
 * @param stride 步长（元素单位）
 * @param sign   RVVFFT_FORWARD 或 RVVFFT_BACKWARD（twiddle已按sign预计算）
 */
void rvvfft_r2_split_f32_scalar(
    float       *real,
    float       *imag,
    const float *tw_r,
    const float *tw_i,
    int          n,
    ptrdiff_t    stride,
    int          sign)
{
    (void)sign;
    int half = n / 2;
    for (int k = 0; k < half; ++k) {
        float *er = real + (ptrdiff_t)k * stride;
        float *ei = imag + (ptrdiff_t)k * stride;
        float *or_ = real + ((ptrdiff_t)k + half) * stride;
        float *oi  = imag + ((ptrdiff_t)k + half) * stride;

        float wr = tw_r[k], wi = tw_i[k];

        /* t = W · O */
        float tr = wr * (*or_) - wi * (*oi);
        float ti = wi * (*or_) + wr * (*oi);

        float e_r = *er, e_i = *ei;
        *er  = e_r + tr;
        *ei  = e_i + ti;
        *or_ = e_r - tr;
        *oi  = e_i - ti;
    }
}

/* ── RVV Intrinsic 实现（分离布局，所有load/store连续） ─────── */
#ifdef RVVFFT_USE_RVV
#include <riscv_vector.h>

/**
 * @brief Radix-2 DIT butterfly，分离布局，VLEN=256 RVV实现。
 *
 * 分离布局下实部/虚部已在独立连续数组，直接vle32加载，
 * 无需vrgather或vlseg，指令效率更高。
 */
void rvvfft_r2_split_f32_vl256(
    float       *real,
    float       *imag,
    const float *tw_r,
    const float *tw_i,
    int          n,
    ptrdiff_t    stride,
    int          sign)
{
    (void)sign;
    int half = n / 2;

    if (stride != 1) {
        rvvfft_r2_split_f32_scalar(real, imag, tw_r, tw_i, n, stride, sign);
        return;
    }

    float *e_real = real;
    float *e_imag = imag;
    float *o_real = real + half;
    float *o_imag = imag + half;

    int remaining = half;
    int offset    = 0;

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)remaining);

        /* 加载twiddle因子 */
        vfloat32m1_t vwr = __riscv_vle32_v_f32m1(tw_r + offset, vl);
        vfloat32m1_t vwi = __riscv_vle32_v_f32m1(tw_i + offset, vl);

        /* 加载实部/虚部（连续访问，无需gather） */
        vfloat32m1_t ve_r = __riscv_vle32_v_f32m1(e_real + offset, vl);
        vfloat32m1_t ve_i = __riscv_vle32_v_f32m1(e_imag + offset, vl);
        vfloat32m1_t vo_r = __riscv_vle32_v_f32m1(o_real + offset, vl);
        vfloat32m1_t vo_i = __riscv_vle32_v_f32m1(o_imag + offset, vl);

        /*
         * 复数乘法 t = W · O：
         *   tr = wr*or - wi*oi  (vfmul + vfnmsac)
         *   ti = wi*or + wr*oi  (vfmul + vfmacc)
         */
        vfloat32m1_t vtr = __riscv_vfmul_vv_f32m1(vwr, vo_r, vl);
        vtr = __riscv_vfnmsac_vv_f32m1(vtr, vwi, vo_i, vl);

        vfloat32m1_t vti = __riscv_vfmul_vv_f32m1(vwi, vo_r, vl);
        vti = __riscv_vfmacc_vv_f32m1(vti, vwr, vo_i, vl);

        /* Butterfly */
        vfloat32m1_t ve_r_out = __riscv_vfadd_vv_f32m1(ve_r, vtr, vl);
        vfloat32m1_t ve_i_out = __riscv_vfadd_vv_f32m1(ve_i, vti, vl);
        vfloat32m1_t vo_r_out = __riscv_vfsub_vv_f32m1(ve_r, vtr, vl);
        vfloat32m1_t vo_i_out = __riscv_vfsub_vv_f32m1(ve_i, vti, vl);

        /* 写回（连续存储） */
        __riscv_vse32_v_f32m1(e_real + offset, ve_r_out, vl);
        __riscv_vse32_v_f32m1(e_imag + offset, ve_i_out, vl);
        __riscv_vse32_v_f32m1(o_real + offset, vo_r_out, vl);
        __riscv_vse32_v_f32m1(o_imag + offset, vo_i_out, vl);

        offset    += (int)vl;
        remaining -= (int)vl;
    }
}

/**
 * @brief Radix-2 DIT butterfly，分离布局，VLEN=1024 RVV实现。
 *
 * 与VLEN=256版本共享算法实现；通过不同的VLEN配置，
 * 编译器会为vsetvl生成不同立即数，达到向量宽度特化效果。
 * 后续优化中可在此版本引入LMUL=4以进一步提高指令并行度。
 */
void rvvfft_r2_split_f32_vl1024(
    float       *real,
    float       *imag,
    const float *tw_r,
    const float *tw_i,
    int          n,
    ptrdiff_t    stride,
    int          sign)
{
    rvvfft_r2_split_f32_vl256(real, imag, tw_r, tw_i, n, stride, sign);
}

#endif /* RVVFFT_USE_RVV */
