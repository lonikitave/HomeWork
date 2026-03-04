/**
 * split.c — 分离(split)布局适配层
 *
 * 分离布局是本库优先的内部执行格式：
 *   real[n] + imag[n] 分别连续存储，RVV向量load全为连续访问。
 *
 * 此文件提供：
 *   1. split布局的缓冲区分配辅助函数
 *   2. 布局描述符（供planner记录布局类型）
 *   3. split → interleaved、interleaved → split 的薄包装
 *      （底层调用 interleaved.c 中的实现）
 */
#include <stdlib.h>
#include <string.h>

#include "../../include/rvvfft_types.h"

/* 声明 interleaved.c 中的函数 */
void rvvfft_deinterleave_f32(const float *interleaved,
                              float *real, float *imag, int n);
void rvvfft_reinterleave_f32(const float *real, const float *imag,
                              float *interleaved, int n);

/**
 * @brief split布局工作缓冲区描述符
 *
 * Planner在选择interleaved布局kernel时，若需要临时转换为split格式，
 * 会分配此描述符管理临时缓冲区生命周期。
 */
typedef struct {
    float   *real_buf;   /**< 临时实部缓冲区 */
    float   *imag_buf;   /**< 临时虚部缓冲区 */
    int      n;          /**< 点数 */
} rvvfft_split_buf_t;

/**
 * @brief 分配split布局临时工作缓冲区。
 *
 * @param n  复数点数
 * @return   描述符指针，失败返回NULL
 */
rvvfft_split_buf_t *rvvfft_split_buf_alloc(int n)
{
    rvvfft_split_buf_t *buf = malloc(sizeof(*buf));
    if (!buf) return NULL;

    /* 对齐分配：实部和虚部紧邻，避免两次malloc开销 */
    size_t sz = (size_t)n * sizeof(float);
    buf->real_buf = malloc(2 * sz);
    if (!buf->real_buf) { free(buf); return NULL; }
    buf->imag_buf = buf->real_buf + n;
    buf->n = n;
    return buf;
}

/**
 * @brief 释放split布局临时工作缓冲区。
 */
void rvvfft_split_buf_free(rvvfft_split_buf_t *buf)
{
    if (!buf) return;
    free(buf->real_buf);
    free(buf);
}

/**
 * @brief interleaved → split转换（薄包装）。
 */
void rvvfft_split_from_interleaved(const float *il, float *re, float *im, int n)
{
    rvvfft_deinterleave_f32(il, re, im, n);
}

/**
 * @brief split → interleaved转换（薄包装）。
 */
void rvvfft_split_to_interleaved(const float *re, const float *im, float *il, int n)
{
    rvvfft_reinterleave_f32(re, im, il, n);
}
