#include "rvvfft.h"
#include <stdlib.h>
#include <string.h>

/* Internal plan structure */
struct rvvfft_plan_s {
    int n;
    int sign;
    unsigned flags;
    int layout;      /* 0 = interleaved, 1 = split */
    void *in;
    void *out;
};

rvvfft_plan rvvfft_plan_dft_1d(int n, rvvfft_complex *in, rvvfft_complex *out,
                                int sign, unsigned flags)
{
    struct rvvfft_plan_s *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n = n; p->sign = sign; p->flags = flags;
    p->layout = 0; p->in = in; p->out = out;
    return p;
}

rvvfft_plan rvvfft_plan_dft_split_1d(int n,
                                      float *ri, float *ii,
                                      float *ro, float *io,
                                      int sign, unsigned flags)
{
    struct rvvfft_plan_s *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n = n; p->sign = sign; p->flags = flags;
    p->layout = 1; p->in = ri; p->out = ro;
    return p;
}

void rvvfft_destroy_plan(rvvfft_plan plan)
{
    free(plan);
}
