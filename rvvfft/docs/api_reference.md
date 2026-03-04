# rvvfft API Reference

**rvvfft** is a high-performance FFT library for self-developed RISC-V RVV CPU chips (VLEN=256 / VLEN=1024).  It follows the FFTW-style plan → execute API.

---

## Quick Start

```c
#include "rvvfft.h"

// 1. Initialise the library (once at startup)
rvvfft_init();

// 2. Allocate aligned buffers
rvvfft_complex *in  = rvvfft_malloc(N * sizeof(rvvfft_complex));
rvvfft_complex *out = rvvfft_malloc(N * sizeof(rvvfft_complex));

// 3. Create a plan
rvvfft_plan p = rvvfft_plan_dft_1d(N, in, out, RVVFFT_FORWARD, RVVFFT_ESTIMATE);

// 4. Fill input and execute
fill_input(in, N);
rvvfft_execute(p);

// 5. Destroy the plan and free buffers
rvvfft_destroy_plan(p);
rvvfft_free(in);
rvvfft_free(out);
rvvfft_cleanup();
```

---

## Initialisation

### `rvvfft_status_t rvvfft_init(void)`

Initialise the library.  Detects hardware VLEN via `csrr vlenb`.  Call once before any other API.

### `void rvvfft_cleanup(void)`

Release all library-wide resources.

---

## Data Types

| Type | Description |
|---|---|
| `rvvfft_complex` | `struct { float r, i; }` — single-precision complex |
| `rvvfft_complex_f64` | `struct { double r, i; }` — double-precision complex |
| `rvvfft_plan` | Opaque plan handle |
| `rvvfft_status_t` | Error code enum |
| `rvvfft_layout_t` | `RVVFFT_LAYOUT_INTERLEAVED` or `RVVFFT_LAYOUT_SPLIT` |

---

## Constants

| Constant | Value | Meaning |
|---|---|---|
| `RVVFFT_FORWARD` | `-1` | Forward transform (e^{-j 2π k/N}) |
| `RVVFFT_BACKWARD` | `+1` | Inverse transform (e^{+j 2π k/N}) |
| `RVVFFT_ESTIMATE` | `0` | Heuristic plan (fast, no benchmarking) |
| `RVVFFT_MEASURE` | `1` | Auto-tuned plan (slower setup, faster execution) |
| `RVVFFT_ALIGN_BYTES` | `128` | Required buffer alignment (= VLEN_1024 / 8) |

---

## Planning Functions

### `rvvfft_plan rvvfft_plan_dft_1d(int n, rvvfft_complex *in, rvvfft_complex *out, int sign, unsigned flags)`

Create a 1D complex-to-complex plan using **interleaved** layout (`[r0,i0,r1,i1,...]`).

- `n` must be a power of 2, ≥ 2.
- `in` and `out` must be `RVVFFT_ALIGN_BYTES`-aligned.
- `in == out` is permitted (in-place transform).

### `rvvfft_plan rvvfft_plan_dft_split_1d(int n, const float *ri, const float *ii, float *ro, float *io, int sign, unsigned flags)`

Create a 1D C2C plan using **split** layout (separate real/imag arrays).  Split layout is recommended for RVV hardware because all vector loads are unit-stride.

### `rvvfft_plan rvvfft_plan_dft_1d_f64(int n, rvvfft_complex_f64 *in, rvvfft_complex_f64 *out, int sign, unsigned flags)`

Double-precision interleaved variant.  *(Phase 1.8)*

Returns `NULL` on error (invalid size, allocation failure, unsupported transform).

---

## Execution

### `void rvvfft_execute(const rvvfft_plan plan)`

Execute the plan.  Thread-safe: the same plan can be executed from multiple threads simultaneously, provided each thread uses distinct I/O buffers.

### `void rvvfft_execute_dft(const rvvfft_plan plan, rvvfft_complex *in, rvvfft_complex *out)`

Execute with different I/O buffers.  Useful for reusing a plan across multiple buffer pairs.

### `void rvvfft_execute_split_dft(const rvvfft_plan plan, const float *ri, const float *ii, float *ro, float *io)`

Split-layout variant of `rvvfft_execute_dft`.

---

## Lifecycle

### `void rvvfft_destroy_plan(rvvfft_plan plan)`

Free all resources associated with a plan.  `NULL` is a no-op.

---

## Wisdom (Plan Caching)

```c
// Save accumulated plan measurements to disk
rvvfft_export_wisdom_to_file("/etc/rvvfft.wisdom");

// Load on next run (avoids re-measurement)
rvvfft_import_wisdom_from_file("/etc/rvvfft.wisdom");

// Discard all cached wisdom
rvvfft_forget_wisdom();
```

Wisdom files are machine-specific (VLEN, frequency, microarchitecture).  They can be distributed with the SDK for a known hardware configuration.

---

## Memory Utilities

### `void *rvvfft_malloc(size_t size)`

Allocate `RVVFFT_ALIGN_BYTES`-aligned memory.

### `void rvvfft_free(void *ptr)`

Free memory allocated by `rvvfft_malloc`.  `NULL` is a no-op.

### `int rvvfft_is_aligned(const void *ptr)`

Return non-zero if `ptr` is properly aligned for rvvfft.

---

## Introspection

### `const char *rvvfft_version(void)`

Return library version string (e.g. `"0.1.0"`).

### `int rvvfft_detected_vlen(void)`

Return the hardware VLEN detected at `rvvfft_init()` (256 or 1024).  Returns 0 before initialisation.

### `double rvvfft_flops(int n)`

Return the standard FLOP count for a length-`n` FFT: `5 * n * log2(n)`.

---

## Data Layout Comparison

| Layout | Memory format | Load efficiency | Interop |
|---|---|---|---|
| Interleaved | `[r0,i0,r1,i1,...]` | Strided load (overhead) | FFTW-compatible |
| Split | `real[]: [r0,r1,...]` + `imag[]: [i0,i1,...]` | Unit-stride (optimal) | Custom |

**Recommendation:** Use split layout for best RVV throughput.  Use interleaved when interfacing with external libraries that expect FFTW-style format.

---

## Error Codes

| Code | Value | Meaning |
|---|---|---|
| `RVVFFT_SUCCESS` | 0 | No error |
| `RVVFFT_ERR_INVALID_SIZE` | -1 | `n` is not a supported power of 2 |
| `RVVFFT_ERR_INVALID_PARAM` | -2 | NULL pointer or invalid flag |
| `RVVFFT_ERR_NO_MEMORY` | -3 | Allocation failure |
| `RVVFFT_ERR_NO_WISDOM` | -4 | Wisdom file not found |
| `RVVFFT_ERR_UNSUPPORTED` | -5 | Transform configuration not yet implemented |
