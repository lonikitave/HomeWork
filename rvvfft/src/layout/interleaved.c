/**
 * interleaved.c — 交错(interleaved)布局适配层
 *
 * 负责在interleaved复数数组与split实/虚分离数组之间做转换，
 * 以及提供供planner使用的布局检查和描述接口。
 *
 * 当用户以interleaved布局调用API时，executor通过此层：
 *   1. 判断是否需要先解交错（de-interleave）到分离格式再执行，
 *      或直接走interleaved的kernel路径。
 *   2. 执行完成后若需要，重新交错（re-interleave）写回。
 */
#include <stdlib.h>
#include <string.h>

#include "../../include/rvvfft_types.h"

/**
 * @brief 将交错复数数组转换为分离实/虚数组（de-interleave）。
 *
 * @param interleaved  输入：[r0,i0,r1,i1,...] 长度2n的float数组
 * @param real         输出实部（长度n）
 * @param imag         输出虚部（长度n）
 * @param n            复数点数
 */
void rvvfft_deinterleave_f32(const float *interleaved,
                              float       *real,
                              float       *imag,
                              int          n)
{
#ifdef RVVFFT_USE_RVV
#include <riscv_vector.h>
    int remaining = n;
    int off = 0;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)remaining);
        /* vlseg2e32: 从交错数组分别加载实部和虚部 */
        vfloat32m1x2_t seg = __riscv_vlseg2e32_v_f32m1x2(
            interleaved + off * 2, vl);
        __riscv_vse32_v_f32m1(real + off,
            __riscv_vget_v_f32m1x2_f32m1(seg, 0), vl);
        __riscv_vse32_v_f32m1(imag + off,
            __riscv_vget_v_f32m1x2_f32m1(seg, 1), vl);
        off       += (int)vl;
        remaining -= (int)vl;
    }
#else
    for (int k = 0; k < n; ++k) {
        real[k] = interleaved[2*k    ];
        imag[k] = interleaved[2*k + 1];
    }
#endif
}

/**
 * @brief 将分离实/虚数组重新交错为复数数组（re-interleave）。
 *
 * @param real         输入实部（长度n）
 * @param imag         输入虚部（长度n）
 * @param interleaved  输出：[r0,i0,r1,i1,...] 长度2n的float数组
 * @param n            复数点数
 */
void rvvfft_reinterleave_f32(const float *real,
                              const float *imag,
                              float       *interleaved,
                              int          n)
{
#ifdef RVVFFT_USE_RVV
#include <riscv_vector.h>
    int remaining = n;
    int off = 0;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)remaining);
        vfloat32m1_t vr = __riscv_vle32_v_f32m1(real + off, vl);
        vfloat32m1_t vi = __riscv_vle32_v_f32m1(imag + off, vl);
        vfloat32m1x2_t seg = __riscv_vcreate_v_f32m1x2(vr, vi);
        __riscv_vsseg2e32_v_f32m1x2(interleaved + off * 2, seg, vl);
        off       += (int)vl;
        remaining -= (int)vl;
    }
#else
    for (int k = 0; k < n; ++k) {
        interleaved[2*k    ] = real[k];
        interleaved[2*k + 1] = imag[k];
    }
#endif
}
