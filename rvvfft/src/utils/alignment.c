/**
 * alignment.c — 内存对齐工具
 *
 * 提供按VLEN/8字节（最大128字节）对齐的内存分配与检查工具，
 * 确保RVV向量load/store指令的性能最优路径。
 */
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "../../include/rvvfft_types.h"

/**
 * @brief 分配按RVVFFT_ALIGN_BYTES对齐的内存。
 *
 * @param size 分配字节数（0时返回NULL）
 * @return 对齐指针，失败返回NULL
 */
void *rvvfft_aligned_malloc(size_t size)
{
    if (size == 0) return NULL;

    void *ptr = NULL;

#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    /* C11标准方式：aligned_alloc要求size是alignment的倍数 */
    size_t aligned_size = RVVFFT_ALIGN_UP(size, RVVFFT_ALIGN_BYTES);
    ptr = aligned_alloc(RVVFFT_ALIGN_BYTES, aligned_size);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&ptr, RVVFFT_ALIGN_BYTES, size) != 0)
        ptr = NULL;
#else
    /* 回退：分配额外空间，手动对齐（需要保存原始指针用于free） */
    size_t total = size + RVVFFT_ALIGN_BYTES + sizeof(void *);
    void *raw = malloc(total);
    if (raw) {
        void **aligned = (void **)(((uintptr_t)raw + sizeof(void *) +
                                    RVVFFT_ALIGN_BYTES - 1) &
                                   ~(uintptr_t)(RVVFFT_ALIGN_BYTES - 1));
        aligned[-1] = raw;
        ptr = aligned;
    }
#endif
    return ptr;
}

/**
 * @brief 释放由rvvfft_aligned_malloc分配的内存。
 *
 * @param ptr 对齐指针（NULL时无操作）
 */
void rvvfft_aligned_free(void *ptr)
{
    if (!ptr) return;

#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || \
    (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
    free(ptr);
#else
    /* 回退模式：取回原始指针 */
    free(((void **)ptr)[-1]);
#endif
}

/**
 * @brief 检查指针是否按RVVFFT_ALIGN_BYTES对齐。
 *
 * @param ptr 待检查指针
 * @return 1表示对齐，0表示未对齐
 */
int rvvfft_is_aligned(const void *ptr)
{
    return ((uintptr_t)ptr & (RVVFFT_ALIGN_BYTES - 1)) == 0;
}

/**
 * @brief 分配并清零对齐内存（类似calloc）。
 *
 * @param count  元素数量
 * @param size   单个元素字节数
 * @return 对齐并清零的指针，失败返回NULL
 */
void *rvvfft_aligned_calloc(size_t count, size_t size)
{
    size_t total = count * size;
    void *ptr = rvvfft_aligned_malloc(total);
    if (ptr) {
        /* 手动清零（aligned_alloc不保证清零） */
        __builtin_memset(ptr, 0, total);
    }
    return ptr;
}
