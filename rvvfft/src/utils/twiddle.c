#include <math.h>
#include <stdlib.h>

/* Precompute twiddle factors W(k,N) = exp(-2πi·k/N), stored interleaved. */
float *rvvfft_twiddle_alloc(int n)
{
    float *table = malloc(2 * n * sizeof(float));
    if (!table) return NULL;
    for (int k = 0; k < n; k++) {
        double angle = -2.0 * M_PI * k / n;
        table[2 * k]     = (float)cos(angle);
        table[2 * k + 1] = (float)sin(angle);
    }
    return table;
}

void rvvfft_twiddle_free(float *table)
{
    free(table);
}
