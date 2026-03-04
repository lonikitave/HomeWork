/**
 * rvvfft_types.h — Type definitions, constants and macros for rvvfft
 *
 * Target: Self-developed RISC-V RVV CPU chips (VLEN=256 / VLEN=1024)
 */

#ifndef RVVFFT_TYPES_H
#define RVVFFT_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Transform direction ─────────────────────────────────────────── */
#define RVVFFT_FORWARD  (-1)
#define RVVFFT_BACKWARD ( 1)

/* ─── Planning flags ──────────────────────────────────────────────── */
#define RVVFFT_ESTIMATE  (0u)      /* Heuristic plan, no measurement   */
#define RVVFFT_MEASURE   (1u << 0) /* Time all candidate plans         */
#define RVVFFT_WISDOM_ONLY (1u << 1) /* Only use cached wisdom         */

/* ─── Data layouts ────────────────────────────────────────────────── */
typedef enum {
    RVVFFT_LAYOUT_INTERLEAVED = 0, /* [r0, i0, r1, i1, ...]           */
    RVVFFT_LAYOUT_SPLIT       = 1  /* real[]: [r0,r1,...] + imag[]     */
} rvvfft_layout_t;

/* ─── Complex number types ────────────────────────────────────────── */
typedef struct { float  r, i; } rvvfft_complex_f32;
typedef struct { double r, i; } rvvfft_complex_f64;

/* Default alias (single precision) */
typedef rvvfft_complex_f32 rvvfft_complex;

/* ─── VLEN selector ───────────────────────────────────────────────── */
typedef enum {
    RVVFFT_VLEN_256  = 256,
    RVVFFT_VLEN_1024 = 1024
} rvvfft_vlen_t;

/* ─── Supported radix values ──────────────────────────────────────── */
typedef enum {
    RVVFFT_RADIX_2  = 2,
    RVVFFT_RADIX_4  = 4,
    RVVFFT_RADIX_8  = 8,
    RVVFFT_RADIX_16 = 16,
    RVVFFT_RADIX_32 = 32
} rvvfft_radix_t;

/* ─── Scalar data types ───────────────────────────────────────────── */
typedef enum {
    RVVFFT_DTYPE_F32 = 0,
    RVVFFT_DTYPE_F64 = 1
} rvvfft_dtype_t;

/* ─── LMUL values ─────────────────────────────────────────────────── */
typedef enum {
    RVVFFT_LMUL_1 = 1,
    RVVFFT_LMUL_2 = 2,
    RVVFFT_LMUL_4 = 4,
    RVVFFT_LMUL_8 = 8
} rvvfft_lmul_t;

/* ─── Opaque plan handle ──────────────────────────────────────────── */
typedef struct rvvfft_plan_s *rvvfft_plan;

/* ─── Error codes ─────────────────────────────────────────────────── */
typedef enum {
    RVVFFT_SUCCESS           =  0,
    RVVFFT_ERR_INVALID_SIZE  = -1,
    RVVFFT_ERR_INVALID_PARAM = -2,
    RVVFFT_ERR_NO_MEMORY     = -3,
    RVVFFT_ERR_NO_WISDOM     = -4,
    RVVFFT_ERR_UNSUPPORTED   = -5
} rvvfft_status_t;

/* ─── Alignment requirement (VLEN/8 bytes, use max) ──────────────── */
#define RVVFFT_ALIGN_BYTES 128   /* 1024/8 = 128, covers both VLENs   */

/* ─── Helper: check if n is a power of 2 ─────────────────────────── */
static inline int rvvfft_is_power_of_2(int n)
{
    return (n > 0) && ((n & (n - 1)) == 0);
}

/* ─── Helper: integer log2 (n must be power of 2) ────────────────── */
static inline int rvvfft_ilog2(int n)
{
    int k = 0;
    while (n > 1) { n >>= 1; ++k; }
    return k;
}

#ifdef __cplusplus
}
#endif

#endif /* RVVFFT_TYPES_H */
