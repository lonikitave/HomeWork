#ifndef RVVFFT_H
#define RVVFFT_H

#include "rvvfft_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── 1-D complex-to-complex FFT (interleaved layout) ───────────────────── */
rvvfft_plan rvvfft_plan_dft_1d(int n,
                                rvvfft_complex *in,
                                rvvfft_complex *out,
                                int sign,
                                unsigned flags);

/* ── 1-D complex-to-complex FFT (split layout) ──────────────────────────── */
rvvfft_plan rvvfft_plan_dft_split_1d(int n,
                                      float *ri, float *ii,
                                      float *ro, float *io,
                                      int sign,
                                      unsigned flags);

/* ── Execute / destroy ──────────────────────────────────────────────────── */
void rvvfft_execute(const rvvfft_plan plan);
void rvvfft_destroy_plan(rvvfft_plan plan);

/* ── Wisdom (plan serialisation) ────────────────────────────────────────── */
int  rvvfft_export_wisdom(const char *path);
int  rvvfft_import_wisdom(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* RVVFFT_H */
