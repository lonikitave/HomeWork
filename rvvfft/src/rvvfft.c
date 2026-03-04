/**
 * rvvfft.c — Library initialisation, teardown and utility functions.
 */

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "rvvfft.h"
#include "rvvfft_types.h"

#define RVVFFT_VERSION_STR "0.1.0"

/* ─── Library state ──────────────────────────────────────────────────────── */

static int   g_initialised = 0;
static int   g_vlen        = 0;   /* hardware VLEN, detected at init */

/* ─── Hardware VLEN detection ─────────────────────────────────────────────── */

/**
 * detect_vlen — Read the hardware VLEN via csrr vlenb.
 *
 * vlenb = VLEN / 8.  Returns 0 if the V extension is not available.
 */
static int detect_vlen(void)
{
#if defined(__riscv_v)
    /* Read vlenb CSR: VLEN in bytes */
    unsigned long vlenb = 0;
    __asm__ volatile ("csrr %0, vlenb" : "=r"(vlenb));
    return (int)(vlenb * 8);
#else
    /* Non-RISC-V host: return a sensible default for build/test purposes */
    return 256;
#endif
}

/* ─── Public API ──────────────────────────────────────────────────────────── */

rvvfft_status_t rvvfft_init(void)
{
    if (g_initialised)
        return RVVFFT_SUCCESS;

    g_vlen = detect_vlen();
    g_initialised = 1;
    return RVVFFT_SUCCESS;
}

void rvvfft_cleanup(void)
{
    g_initialised = 0;
    g_vlen        = 0;
}

const char *rvvfft_version(void)
{
    return RVVFFT_VERSION_STR;
}

int rvvfft_detected_vlen(void)
{
    return g_vlen;
}

double rvvfft_flops(int n)
{
    if (n <= 1) return 0.0;
    return 5.0 * (double)n * log2((double)n);
}

/* ─── Wisdom stubs (implemented in wisdom.c in Phase 3) ─────────────────── */

rvvfft_status_t rvvfft_export_wisdom_to_file(const char *path)
{
    (void)path;
    return RVVFFT_SUCCESS;  /* no-op until Phase 3 */
}

rvvfft_status_t rvvfft_import_wisdom_from_file(const char *path)
{
    (void)path;
    return RVVFFT_ERR_NO_WISDOM;  /* no wisdom yet */
}

void rvvfft_forget_wisdom(void)
{
    /* no-op until Phase 3 */
}
