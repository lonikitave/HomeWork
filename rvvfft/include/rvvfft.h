/**
 * rvvfft.h — Public API for the RVV FFT library
 *
 * Supports:
 *   - 1D complex-to-complex (C2C) transforms, single and double precision
 *   - Interleaved and split data layouts
 *   - Forward (sign = -1) and inverse (sign = +1) transforms
 *   - ESTIMATE (heuristic) and MEASURE (auto-tuned) planning modes
 *   - Wisdom file I/O for plan persistence
 *
 * Target hardware: Self-developed RISC-V RVV CPU chips (VLEN=256 / VLEN=1024)
 *
 * Usage example (interleaved, single precision):
 *
 *   #include "rvvfft.h"
 *
 *   rvvfft_complex in[N], out[N];
 *   // ... fill in[] ...
 *   rvvfft_plan p = rvvfft_plan_dft_1d(N, in, out, RVVFFT_FORWARD,
 *                                       RVVFFT_ESTIMATE);
 *   rvvfft_execute(p);
 *   rvvfft_destroy_plan(p);
 *
 * Usage example (split layout, single precision):
 *
 *   float ri[N], ii[N], ro[N], io[N];
 *   // ... fill ri[], ii[] ...
 *   rvvfft_plan p = rvvfft_plan_dft_split_1d(N, ri, ii, ro, io,
 *                                             RVVFFT_FORWARD, RVVFFT_ESTIMATE);
 *   rvvfft_execute(p);
 *   rvvfft_destroy_plan(p);
 */

#ifndef RVVFFT_H
#define RVVFFT_H

#include "rvvfft_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * rvvfft_malloc - Allocate RVVFFT_ALIGN_BYTES-aligned memory.
 *
 * @size: Number of bytes to allocate.
 * Returns an aligned pointer, or NULL on failure or if size == 0.
 */
void *rvvfft_malloc(size_t size);

/**
 * rvvfft_free - Free memory allocated by rvvfft_malloc().
 *
 * @ptr may be NULL (no-op).
 */
void rvvfft_free(void *ptr);

/**
 * rvvfft_is_aligned - Return non-zero if ptr meets the RVVFFT_ALIGN_BYTES
 *                     alignment requirement.
 */
int rvvfft_is_aligned(const void *ptr);

/* ═══════════════════════════════════════════════════════════════════
 * Initialisation / Teardown
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_init - Initialise the library (call once at startup).
 *
 * Detects hardware VLEN and sets up internal singleton state.
 * Thread-safe after the first call completes.
 *
 * Returns RVVFFT_SUCCESS or an error code.
 */
rvvfft_status_t rvvfft_init(void);

/**
 * rvvfft_cleanup - Release all library-wide resources.
 */
void rvvfft_cleanup(void);

/* ═══════════════════════════════════════════════════════════════════
 * 1-D Complex-to-Complex — Interleaved layout (f32)
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_plan_dft_1d - Create a 1-D C2C FFT plan (single precision,
 *                      interleaved [r0,i0,r1,i1,...] layout).
 *
 * @n:    Transform length (must be a power of 2, n >= 2).
 * @in:   Input  array of n complex samples (user-allocated, RVVFFT_ALIGN_BYTES).
 * @out:  Output array of n complex samples (may equal @in for in-place).
 * @sign: RVVFFT_FORWARD (-1) or RVVFFT_BACKWARD (+1).
 * @flags: RVVFFT_ESTIMATE | RVVFFT_MEASURE | RVVFFT_WISDOM_ONLY.
 *
 * Returns a non-NULL plan on success, NULL on failure.
 */
rvvfft_plan rvvfft_plan_dft_1d(int n,
                                rvvfft_complex *in,
                                rvvfft_complex *out,
                                int sign,
                                unsigned flags);

/* ═══════════════════════════════════════════════════════════════════
 * 1-D Complex-to-Complex — Split layout (f32)
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_plan_dft_split_1d - Create a 1-D C2C FFT plan (single precision,
 *                             split real/imag layout).
 *
 * @n:    Transform length (must be a power of 2, n >= 2).
 * @ri:   Input  real  part array (n floats, RVVFFT_ALIGN_BYTES aligned).
 * @ii:   Input  imag  part array (n floats, RVVFFT_ALIGN_BYTES aligned).
 * @ro:   Output real  part array (may equal @ri for in-place).
 * @io:   Output imag  part array (may equal @ii for in-place).
 * @sign: RVVFFT_FORWARD (-1) or RVVFFT_BACKWARD (+1).
 * @flags: Planning flags (same as rvvfft_plan_dft_1d).
 *
 * Returns a non-NULL plan on success, NULL on failure.
 */
rvvfft_plan rvvfft_plan_dft_split_1d(int n,
                                      const float *ri, const float *ii,
                                      float *ro,       float *io,
                                      int sign,
                                      unsigned flags);

/* ═══════════════════════════════════════════════════════════════════
 * 1-D Complex-to-Complex — Interleaved layout (f64)
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_plan_dft_1d_f64 - Double-precision interleaved 1-D C2C plan.
 *
 * Parameters mirror rvvfft_plan_dft_1d() but operate on f64 data.
 */
rvvfft_plan rvvfft_plan_dft_1d_f64(int n,
                                    rvvfft_complex_f64 *in,
                                    rvvfft_complex_f64 *out,
                                    int sign,
                                    unsigned flags);

/* ═══════════════════════════════════════════════════════════════════
 * Execution
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_execute - Execute a previously created FFT plan.
 *
 * Thread-safe: the same plan may be executed concurrently from multiple
 * threads provided that each thread uses distinct input/output buffers.
 * The plan itself is read-only after creation.
 *
 * @plan: Non-NULL plan returned by one of the planning functions.
 */
void rvvfft_execute(const rvvfft_plan plan);

/**
 * rvvfft_execute_dft - Execute an interleaved plan with different I/O buffers.
 *
 * Useful when the same plan is reused across different buffer pairs.
 * Buffer size and alignment must match those used during planning.
 */
void rvvfft_execute_dft(const rvvfft_plan plan,
                         rvvfft_complex *in,
                         rvvfft_complex *out);

/**
 * rvvfft_execute_split_dft - Execute a split plan with different I/O buffers.
 */
void rvvfft_execute_split_dft(const rvvfft_plan plan,
                               const float *ri, const float *ii,
                               float *ro,       float *io);

/* ═══════════════════════════════════════════════════════════════════
 * Plan lifecycle
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_destroy_plan - Free all resources associated with a plan.
 *
 * @plan may be NULL (no-op).
 */
void rvvfft_destroy_plan(rvvfft_plan plan);

/* ═══════════════════════════════════════════════════════════════════
 * Wisdom (plan caching)
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_export_wisdom_to_file - Serialise all accumulated wisdom to disk.
 *
 * @path: File path (will be created/overwritten).
 * Returns RVVFFT_SUCCESS or an error code.
 */
rvvfft_status_t rvvfft_export_wisdom_to_file(const char *path);

/**
 * rvvfft_import_wisdom_from_file - Load previously exported wisdom.
 *
 * @path: File path to load from.
 * Returns RVVFFT_SUCCESS or RVVFFT_ERR_NO_WISDOM if the file is absent.
 */
rvvfft_status_t rvvfft_import_wisdom_from_file(const char *path);

/**
 * rvvfft_forget_wisdom - Discard all cached wisdom.
 */
void rvvfft_forget_wisdom(void);

/* ═══════════════════════════════════════════════════════════════════
 * Utility / introspection
 * ═══════════════════════════════════════════════════════════════════ */

/**
 * rvvfft_version - Return library version string (e.g. "1.0.0").
 */
const char *rvvfft_version(void);

/**
 * rvvfft_detected_vlen - Return the hardware VLEN detected at init time
 *                        (256 or 1024).  Returns 0 before rvvfft_init().
 */
int rvvfft_detected_vlen(void);

/**
 * rvvfft_flops - Return the number of floating-point operations for a
 *               length-n FFT (= 5 * n * log2(n), standard formula).
 */
double rvvfft_flops(int n);

#ifdef __cplusplus
}
#endif

#endif /* RVVFFT_H */
