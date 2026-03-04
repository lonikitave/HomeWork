/**
 * bitrev.c — Bit-reversal permutation
 *
 * FFT Cooley-Tukey算法结束后，若采用DIT（时域抽取）方式，
 * 需要对输入（或输出）按bit-reversal顺序重排。
 *
 * 本模块提供：
 *   1. 标量通用实现（VL-agnostic，适用于任意VLEN）
 *   2. 原位(in-place)复数数组重排（interleaved和split两种布局）
 *   3. 位序逆转索引预计算（供planner缓存，executor直接使用）
 */
#include <stdlib.h>
#include <string.h>

#include "../../include/rvvfft_types.h"

/**
 * @brief 计算整数x的log2（x必须是2的幂）。
 */
static inline int ilog2(int x)
{
    int r = 0;
    while (x > 1) { x >>= 1; ++r; }
    return r;
}

/**
 * @brief 将整数x按bits位宽做bit-reversal。
 *
 * 例：bits=4, x=0b0110 → 0b0110（6）→ 反转 → 0b0110（6）
 *    bits=4, x=0b0001 → 反转 → 0b1000（8）
 */
static inline int bit_reverse(int x, int bits)
{
    int y = 0;
    for (int i = 0; i < bits; ++i) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

/**
 * @brief 为长度n的FFT预计算bit-reversal置换表。
 *
 * @param n    DFT长度（必须是2的幂）
 * @param perm 输出置换表，调用者负责分配（长度≥n的int数组）
 */
void rvvfft_bitrev_table(int n, int *perm)
{
    int bits = ilog2(n);
    for (int i = 0; i < n; ++i)
        perm[i] = bit_reverse(i, bits);
}

/**
 * @brief 对交错复数数组执行原位bit-reversal置换（f32）。
 *
 * @param data   长度为n的rvvfft_complex_f32数组
 * @param n      DFT长度（2的幂）
 * @param perm   由rvvfft_bitrev_table预计算的置换表（长度n）
 */
void rvvfft_bitrev_inplace_f32(rvvfft_complex_f32 *data, int n, const int *perm)
{
    for (int i = 0; i < n; ++i) {
        int j = perm[i];
        if (j > i) {
            rvvfft_complex_f32 tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

/**
 * @brief 对分离布局实数/虚数数组执行原位bit-reversal置换（f32）。
 *
 * @param real  实部数组（长度n）
 * @param imag  虚部数组（长度n）
 * @param n     DFT长度
 * @param perm  置换表
 */
void rvvfft_bitrev_inplace_split_f32(float *real, float *imag,
                                      int n, const int *perm)
{
    for (int i = 0; i < n; ++i) {
        int j = perm[i];
        if (j > i) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }
}

/**
 * @brief 将交错布局数组按bit-reversal顺序复制到输出（out-of-place，f32）。
 *
 * @param in   输入交错复数数组
 * @param out  输出交错复数数组（与in不重叠）
 * @param n    DFT长度
 * @param perm 置换表
 */
void rvvfft_bitrev_copy_f32(const rvvfft_complex_f32 *in,
                             rvvfft_complex_f32       *out,
                             int n, const int *perm)
{
    for (int i = 0; i < n; ++i)
        out[perm[i]] = in[i];
}
