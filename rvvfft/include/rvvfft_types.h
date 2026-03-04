#ifndef RVVFFT_TYPES_H
#define RVVFFT_TYPES_H

#include <stdint.h>

/* ── Data types ─────────────────────────────────────────────────────────── */

typedef struct { float  re, im; } rvvfft_complex;
typedef struct { double re, im; } rvvfft_complexd;

/* ── Transform direction ────────────────────────────────────────────────── */
#define RVVFFT_FORWARD  (-1)
#define RVVFFT_BACKWARD ( 1)

/* ── Planning flags ─────────────────────────────────────────────────────── */
#define RVVFFT_ESTIMATE  (0u)      /* heuristic plan selection */
#define RVVFFT_MEASURE   (1u << 0) /* benchmark all candidate plans */

/* ── Internal plan handle (opaque) ─────────────────────────────────────── */
typedef struct rvvfft_plan_s *rvvfft_plan;

#endif /* RVVFFT_TYPES_H */
