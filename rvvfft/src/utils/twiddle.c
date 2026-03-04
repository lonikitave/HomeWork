/**
 * twiddle.c — Twiddle factor表生成与管理
 *
 * 预计算 W(k,N) = exp(-2πi·k/N) 的实部和虚部，
 * 按VLEN对齐存储，供butterfly kernel直接向量化加载。
 *
 * 布局：tw_real[k] = cos(2π·k/N), tw_imag[k] = -sin(2π·k/N)
 * 支持f32和f64两种精度，各独立缓存。
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/rvvfft_types.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* ── 内部twiddle缓存条目 ──────────────────────────────────── */
typedef struct tw_cache_entry_s {
    int     n;            /**< DFT长度 */
    int     sign;         /**< RVVFFT_FORWARD or RVVFFT_BACKWARD */
    float  *tw_real_f32;  /**< 对齐的实部数组（长度 n/2） */
    float  *tw_imag_f32;  /**< 对齐的虚部数组（长度 n/2） */
    double *tw_real_f64;  /**< f64版本 */
    double *tw_imag_f64;
    struct tw_cache_entry_s *next;
} tw_cache_entry_t;

static tw_cache_entry_t *g_tw_cache = NULL;

/* ── 对齐内存分配 ─────────────────────────────────────────── */
static void *aligned_alloc_rvv(size_t size)
{
    void *ptr = NULL;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&ptr, RVVFFT_ALIGN_BYTES, size) != 0)
        return NULL;
#else
    ptr = malloc(size + RVVFFT_ALIGN_BYTES);
    if (!ptr) return NULL;
    uintptr_t aligned = RVVFFT_ALIGN_UP(ptr, RVVFFT_ALIGN_BYTES);
    /* 简单实现：存在内存泄漏风险，生产版本应使用posix_memalign */
    ptr = (void *)aligned;
#endif
    return ptr;
}

/**
 * @brief 生成并缓存长度n的twiddle factor表（f32精度）。
 *
 * @param n    DFT长度（2的幂）
 * @param sign RVVFFT_FORWARD 或 RVVFFT_BACKWARD
 * @param[out] out_real  输出实部指针（指向缓存，调用者不得释放）
 * @param[out] out_imag  输出虚部指针
 * @return RVVFFT_SUCCESS 或错误码
 */
rvvfft_status_t rvvfft_twiddle_get_f32(int n, int sign,
                                         const float **out_real,
                                         const float **out_imag)
{
    if (!RVVFFT_IS_POW2(n) || !out_real || !out_imag)
        return RVVFFT_ERR_INVALID_ARG;

    /* 查找缓存 */
    for (tw_cache_entry_t *e = g_tw_cache; e; e = e->next) {
        if (e->n == n && e->sign == sign && e->tw_real_f32) {
            *out_real = e->tw_real_f32;
            *out_imag = e->tw_imag_f32;
            return RVVFFT_SUCCESS;
        }
    }

    /* 新建条目 */
    tw_cache_entry_t *entry = calloc(1, sizeof(*entry));
    if (!entry) return RVVFFT_ERR_NO_MEMORY;

    int half_n = n / 2;  /* butterfly每级需要n/2个twiddle */
    entry->n    = n;
    entry->sign = sign;
    entry->tw_real_f32 = aligned_alloc_rvv((size_t)half_n * sizeof(float));
    entry->tw_imag_f32 = aligned_alloc_rvv((size_t)half_n * sizeof(float));
    if (!entry->tw_real_f32 || !entry->tw_imag_f32) {
        free(entry->tw_real_f32);
        free(entry->tw_imag_f32);
        free(entry);
        return RVVFFT_ERR_NO_MEMORY;
    }

    /* 计算 W(k,N) = exp(sign * 2πi·k/N) */
    double theta_base = sign * 2.0 * M_PI / (double)n;
    for (int k = 0; k < half_n; ++k) {
        double theta = theta_base * k;
        entry->tw_real_f32[k] = (float)cos(theta);
        entry->tw_imag_f32[k] = (float)sin(theta);
    }

    entry->next = g_tw_cache;
    g_tw_cache  = entry;

    *out_real = entry->tw_real_f32;
    *out_imag = entry->tw_imag_f32;
    return RVVFFT_SUCCESS;
}

/**
 * @brief 生成并缓存长度n的twiddle factor表（f64精度）。
 */
rvvfft_status_t rvvfft_twiddle_get_f64(int n, int sign,
                                         const double **out_real,
                                         const double **out_imag)
{
    if (!RVVFFT_IS_POW2(n) || !out_real || !out_imag)
        return RVVFFT_ERR_INVALID_ARG;

    for (tw_cache_entry_t *e = g_tw_cache; e; e = e->next) {
        if (e->n == n && e->sign == sign && e->tw_real_f64) {
            *out_real = e->tw_real_f64;
            *out_imag = e->tw_imag_f64;
            return RVVFFT_SUCCESS;
        }
    }

    tw_cache_entry_t *entry = calloc(1, sizeof(*entry));
    if (!entry) return RVVFFT_ERR_NO_MEMORY;

    int half_n = n / 2;
    entry->n    = n;
    entry->sign = sign;
    entry->tw_real_f64 = aligned_alloc_rvv((size_t)half_n * sizeof(double));
    entry->tw_imag_f64 = aligned_alloc_rvv((size_t)half_n * sizeof(double));
    if (!entry->tw_real_f64 || !entry->tw_imag_f64) {
        free(entry->tw_real_f64);
        free(entry->tw_imag_f64);
        free(entry);
        return RVVFFT_ERR_NO_MEMORY;
    }

    double theta_base = sign * 2.0 * M_PI / (double)n;
    for (int k = 0; k < half_n; ++k) {
        double theta = theta_base * k;
        entry->tw_real_f64[k] = cos(theta);
        entry->tw_imag_f64[k] = sin(theta);
    }

    entry->next = g_tw_cache;
    g_tw_cache  = entry;

    *out_real = entry->tw_real_f64;
    *out_imag = entry->tw_imag_f64;
    return RVVFFT_SUCCESS;
}

/**
 * @brief 释放所有已缓存的twiddle factor表。
 */
void rvvfft_twiddle_free_all(void)
{
    tw_cache_entry_t *e = g_tw_cache;
    while (e) {
        tw_cache_entry_t *next = e->next;
        free(e->tw_real_f32);
        free(e->tw_imag_f32);
        free(e->tw_real_f64);
        free(e->tw_imag_f64);
        free(e);
        e = next;
    }
    g_tw_cache = NULL;
}
