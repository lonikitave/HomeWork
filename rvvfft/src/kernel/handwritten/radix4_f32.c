/**
 * radix4_f32.c — Radix-4 butterfly kernel（interleaved + split, f32）
 *
 * Radix-4 DIT butterfly（每次处理4个子问题）：
 *   设 t0=X0, t1=W^k·X1, t2=W^{2k}·X2, t3=W^{3k}·X3，则：
 *   X0' =  t0 + t1 + t2 + t3
 *   X1' =  t0 - j*t1 - t2 + j*t3    (FORWARD, j=sqrt(-1))
 *   X2' =  t0 - t1 + t2 - t3
 *   X3' =  t0 + j*t1 - t2 - j*t3
 *
 * Radix-4的优势：每个butterfly只需3次复数乘法（相比radix-2的1次/butterfly），
 * 但每次处理4×更多点，使每点乘法次数从1降到3/4，减少25%乘法开销。
 * 同时减少pass数量，降低内存traffic。
 *
 * 提供：标量参考实现 + RVV向量化实现（VLEN=256/1024）
 * 两种布局：interleaved（函数后缀_il）和split（函数后缀_sp）
 */
#include <stddef.h>
#include <math.h>

#include "../../../include/rvvfft_types.h"

/* ══════════════════════════════════════════════════════════
 * 交错布局 (Interleaved) — 标量参考实现
 * ══════════════════════════════════════════════════════════ */

/**
 * @brief Radix-4 DIT butterfly，交错布局，纯C标量实现。
 *
 * @param data   交错复数数组（in-place）
 * @param tw1r   W^k  实部表（长度≥n/4）
 * @param tw1i   W^k  虚部表
 * @param tw2r   W^{2k} 实部表
 * @param tw2i   W^{2k} 虚部表
 * @param tw3r   W^{3k} 实部表
 * @param tw3i   W^{3k} 虚部表
 * @param n      本级DFT长度（必须是4的倍数）
 * @param stride 步长（复数元素单位）
 * @param sign   RVVFFT_FORWARD 或 RVVFFT_BACKWARD
 */
void rvvfft_r4_interleaved_f32_scalar(
    rvvfft_complex_f32 *data,
    const float *tw1r, const float *tw1i,
    const float *tw2r, const float *tw2i,
    const float *tw3r, const float *tw3i,
    int          n,
    ptrdiff_t    stride,
    int          sign)
{
    (void)sign;
    int quarter = n / 4;

    /* 乘法因子±j（FORWARD: -j；twiddle符号已由预计算处理） */
    for (int k = 0; k < quarter; ++k) {
        rvvfft_complex_f32 *x0 = data + (ptrdiff_t)(k             ) * stride;
        rvvfft_complex_f32 *x1 = data + (ptrdiff_t)(k +   quarter) * stride;
        rvvfft_complex_f32 *x2 = data + (ptrdiff_t)(k + 2*quarter) * stride;
        rvvfft_complex_f32 *x3 = data + (ptrdiff_t)(k + 3*quarter) * stride;

        /* t1 = W^k · x1 */
        float t1r = tw1r[k]*x1->r - tw1i[k]*x1->i;
        float t1i = tw1i[k]*x1->r + tw1r[k]*x1->i;
        /* t2 = W^{2k} · x2 */
        float t2r = tw2r[k]*x2->r - tw2i[k]*x2->i;
        float t2i = tw2i[k]*x2->r + tw2r[k]*x2->i;
        /* t3 = W^{3k} · x3 */
        float t3r = tw3r[k]*x3->r - tw3i[k]*x3->i;
        float t3i = tw3i[k]*x3->r + tw3r[k]*x3->i;

        float t0r = x0->r, t0i = x0->i;

        /* 4点DFT蝶形运算 */
        /* 中间量：a = t0+t2, b = t0-t2, c = t1+t3, d = t1-t3 */
        float ar = t0r + t2r, ai = t0i + t2i;
        float br = t0r - t2r, bi = t0i - t2i;
        float cr = t1r + t3r, ci = t1i + t3i;
        float dr = t1r - t3r, di = t1i - t3i;

        /*
         * FORWARD (sign=-1):
         *   X0 = a + c        X1 = b + j·d  (j·d = [-di, dr])
         *   X2 = a - c        X3 = b - j·d
         *
         * BACKWARD (sign=+1):
         *   X1 = b - j·d,  X3 = b + j·d
         * （sign已由twiddle预计算处理，此处统一用+j约定）
         */
        x0->r = ar + cr;   x0->i = ai + ci;
        x1->r = br - di;   x1->i = bi + dr;   /* +j·d */
        x2->r = ar - cr;   x2->i = ai - ci;
        x3->r = br + di;   x3->i = bi - dr;   /* -j·d */
    }
}

/* ══════════════════════════════════════════════════════════
 * 分离布局 (Split) — 标量参考实现
 * ══════════════════════════════════════════════════════════ */

/**
 * @brief Radix-4 DIT butterfly，分离布局，纯C标量实现。
 */
void rvvfft_r4_split_f32_scalar(
    float       *real,
    float       *imag,
    const float *tw1r, const float *tw1i,
    const float *tw2r, const float *tw2i,
    const float *tw3r, const float *tw3i,
    int          n,
    ptrdiff_t    stride,
    int          sign)
{
    (void)sign;
    int quarter = n / 4;

    for (int k = 0; k < quarter; ++k) {
        float *r0 = real + (ptrdiff_t)(k             ) * stride;
        float *i0 = imag + (ptrdiff_t)(k             ) * stride;
        float *r1 = real + (ptrdiff_t)(k +   quarter) * stride;
        float *i1 = imag + (ptrdiff_t)(k +   quarter) * stride;
        float *r2 = real + (ptrdiff_t)(k + 2*quarter) * stride;
        float *i2 = imag + (ptrdiff_t)(k + 2*quarter) * stride;
        float *r3 = real + (ptrdiff_t)(k + 3*quarter) * stride;
        float *i3 = imag + (ptrdiff_t)(k + 3*quarter) * stride;

        float t1r = tw1r[k]*(*r1) - tw1i[k]*(*i1);
        float t1i = tw1i[k]*(*r1) + tw1r[k]*(*i1);
        float t2r = tw2r[k]*(*r2) - tw2i[k]*(*i2);
        float t2i = tw2i[k]*(*r2) + tw2r[k]*(*i2);
        float t3r = tw3r[k]*(*r3) - tw3i[k]*(*i3);
        float t3i = tw3i[k]*(*r3) + tw3r[k]*(*i3);

        float t0r = *r0, t0i = *i0;

        float ar = t0r + t2r, ai = t0i + t2i;
        float br = t0r - t2r, bi = t0i - t2i;
        float cr = t1r + t3r, ci = t1i + t3i;
        float dr = t1r - t3r, di = t1i - t3i;

        *r0 = ar + cr;   *i0 = ai + ci;
        *r1 = br - di;   *i1 = bi + dr;
        *r2 = ar - cr;   *i2 = ai - ci;
        *r3 = br + di;   *i3 = bi - dr;
    }
}

/* ── RVV 向量化实现 ────────────────────────────────────────── */
#ifdef RVVFFT_USE_RVV
#include <riscv_vector.h>

/**
 * @brief Radix-4 DIT butterfly，分离布局，RVV实现（VLEN=256/1024通用）。
 *
 * 分离布局下连续load，无需gather，向量化效率最优。
 * 每次循环处理vl个butterfly（VLEN=256时vl≤8，VLEN=1024时vl≤32）。
 */
void rvvfft_r4_split_f32_rvv(
    float       *real,
    float       *imag,
    const float *tw1r, const float *tw1i,
    const float *tw2r, const float *tw2i,
    const float *tw3r, const float *tw3i,
    int          n,
    ptrdiff_t    stride,
    int          sign)
{
    (void)sign;
    int quarter = n / 4;

    if (stride != 1) {
        rvvfft_r4_split_f32_scalar(real, imag,
            tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, n, stride, sign);
        return;
    }

    float *r0 = real,             *i0 = imag;
    float *r1 = real + quarter,   *i1 = imag + quarter;
    float *r2 = real + 2*quarter, *i2 = imag + 2*quarter;
    float *r3 = real + 3*quarter, *i3 = imag + 3*quarter;

    int remaining = quarter, off = 0;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)remaining);

        vfloat32m1_t v1r = __riscv_vle32_v_f32m1(tw1r + off, vl);
        vfloat32m1_t v1i = __riscv_vle32_v_f32m1(tw1i + off, vl);
        vfloat32m1_t v2r = __riscv_vle32_v_f32m1(tw2r + off, vl);
        vfloat32m1_t v2i = __riscv_vle32_v_f32m1(tw2i + off, vl);
        vfloat32m1_t v3r = __riscv_vle32_v_f32m1(tw3r + off, vl);
        vfloat32m1_t v3i = __riscv_vle32_v_f32m1(tw3i + off, vl);

        vfloat32m1_t vr0 = __riscv_vle32_v_f32m1(r0 + off, vl);
        vfloat32m1_t vi0 = __riscv_vle32_v_f32m1(i0 + off, vl);
        vfloat32m1_t vr1 = __riscv_vle32_v_f32m1(r1 + off, vl);
        vfloat32m1_t vi1 = __riscv_vle32_v_f32m1(i1 + off, vl);
        vfloat32m1_t vr2 = __riscv_vle32_v_f32m1(r2 + off, vl);
        vfloat32m1_t vi2 = __riscv_vle32_v_f32m1(i2 + off, vl);
        vfloat32m1_t vr3 = __riscv_vle32_v_f32m1(r3 + off, vl);
        vfloat32m1_t vi3 = __riscv_vle32_v_f32m1(i3 + off, vl);

        /* t1 = W1 * X1 */
        vfloat32m1_t vt1r = __riscv_vfmul_vv_f32m1(v1r, vr1, vl);
        vt1r = __riscv_vfnmsac_vv_f32m1(vt1r, v1i, vi1, vl);
        vfloat32m1_t vt1i = __riscv_vfmul_vv_f32m1(v1i, vr1, vl);
        vt1i = __riscv_vfmacc_vv_f32m1(vt1i, v1r, vi1, vl);

        /* t2 = W2 * X2 */
        vfloat32m1_t vt2r = __riscv_vfmul_vv_f32m1(v2r, vr2, vl);
        vt2r = __riscv_vfnmsac_vv_f32m1(vt2r, v2i, vi2, vl);
        vfloat32m1_t vt2i = __riscv_vfmul_vv_f32m1(v2i, vr2, vl);
        vt2i = __riscv_vfmacc_vv_f32m1(vt2i, v2r, vi2, vl);

        /* t3 = W3 * X3 */
        vfloat32m1_t vt3r = __riscv_vfmul_vv_f32m1(v3r, vr3, vl);
        vt3r = __riscv_vfnmsac_vv_f32m1(vt3r, v3i, vi3, vl);
        vfloat32m1_t vt3i = __riscv_vfmul_vv_f32m1(v3i, vr3, vl);
        vt3i = __riscv_vfmacc_vv_f32m1(vt3i, v3r, vi3, vl);

        /* a = t0+t2, b = t0-t2, c = t1+t3, d = t1-t3 */
        vfloat32m1_t var = __riscv_vfadd_vv_f32m1(vr0, vt2r, vl);
        vfloat32m1_t vai = __riscv_vfadd_vv_f32m1(vi0, vt2i, vl);
        vfloat32m1_t vbr = __riscv_vfsub_vv_f32m1(vr0, vt2r, vl);
        vfloat32m1_t vbi = __riscv_vfsub_vv_f32m1(vi0, vt2i, vl);
        vfloat32m1_t vcr = __riscv_vfadd_vv_f32m1(vt1r, vt3r, vl);
        vfloat32m1_t vci = __riscv_vfadd_vv_f32m1(vt1i, vt3i, vl);
        vfloat32m1_t vdr = __riscv_vfsub_vv_f32m1(vt1r, vt3r, vl);
        vfloat32m1_t vdi = __riscv_vfsub_vv_f32m1(vt1i, vt3i, vl);

        /* X0=a+c, X1=b+j·d, X2=a-c, X3=b-j·d */
        __riscv_vse32_v_f32m1(r0+off, __riscv_vfadd_vv_f32m1(var, vcr, vl), vl);
        __riscv_vse32_v_f32m1(i0+off, __riscv_vfadd_vv_f32m1(vai, vci, vl), vl);
        __riscv_vse32_v_f32m1(r1+off, __riscv_vfsub_vv_f32m1(vbr, vdi, vl), vl);
        __riscv_vse32_v_f32m1(i1+off, __riscv_vfadd_vv_f32m1(vbi, vdr, vl), vl);
        __riscv_vse32_v_f32m1(r2+off, __riscv_vfsub_vv_f32m1(var, vcr, vl), vl);
        __riscv_vse32_v_f32m1(i2+off, __riscv_vfsub_vv_f32m1(vai, vci, vl), vl);
        __riscv_vse32_v_f32m1(r3+off, __riscv_vfadd_vv_f32m1(vbr, vdi, vl), vl);
        __riscv_vse32_v_f32m1(i3+off, __riscv_vfsub_vv_f32m1(vbi, vdr, vl), vl);

        off       += (int)vl;
        remaining -= (int)vl;
    }
}

#endif /* RVVFFT_USE_RVV */
