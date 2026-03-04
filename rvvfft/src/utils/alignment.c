#include <stdlib.h>
#include <stdint.h>

/* Allocate n bytes aligned to align bytes (align must be a power of two). */
void *rvvfft_aligned_alloc(size_t align, size_t n)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, n) != 0)
        return NULL;
    return ptr;
}

void rvvfft_aligned_free(void *ptr)
{
    free(ptr);
}
