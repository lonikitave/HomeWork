#include <stdint.h>

/* Bit-reversal permutation for arrays of length n (n must be a power of two). */
void rvvfft_bitrev(float *re, float *im, int n)
{
    int bits = 0;
    for (int tmp = n >> 1; tmp; tmp >>= 1) bits++;

    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int b = 0; b < bits; b++)
            j |= ((i >> b) & 1) << (bits - 1 - b);
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
}
