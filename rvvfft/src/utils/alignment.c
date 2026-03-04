/**
 * alignment.c — Memory alignment utilities for rvvfft.
 *
 * All data buffers passed to rvvfft must be aligned to RVVFFT_ALIGN_BYTES
 * (128 bytes, which covers VLEN=1024/8).  These helpers make it easy to
 * allocate correctly aligned buffers.
 */

#include <stdlib.h>
#include <string.h>
#include "rvvfft_types.h"

/**
 * rvvfft_malloc — Allocate an aligned buffer of *size* bytes.
 *
 * The returned pointer is aligned to RVVFFT_ALIGN_BYTES.
 * Free with rvvfft_free().
 *
 * Returns NULL on allocation failure or if size == 0.
 */
void *rvvfft_malloc(size_t size)
{
    if (size == 0)
        return NULL;
    /* Round up size to the alignment boundary so that the allocator
     * does not place the next allocation inside our padding. */
    size_t aligned_size = (size + RVVFFT_ALIGN_BYTES - 1)
                          & ~(size_t)(RVVFFT_ALIGN_BYTES - 1);
    return aligned_alloc(RVVFFT_ALIGN_BYTES, aligned_size);
}

/**
 * rvvfft_free — Release a buffer allocated by rvvfft_malloc().
 *
 * ptr may be NULL (no-op).
 */
void rvvfft_free(void *ptr)
{
    free(ptr);
}

/**
 * rvvfft_is_aligned — Return non-zero if ptr is RVVFFT_ALIGN_BYTES-aligned.
 */
int rvvfft_is_aligned(const void *ptr)
{
    return ((uintptr_t)ptr & (RVVFFT_ALIGN_BYTES - 1)) == 0;
}
