#!/usr/bin/env python3
"""
gen_golden.py — Generate golden FFT reference data using NumPy.

Outputs binary files that the C correctness test driver reads:
  tests/golden/fft_<n>_f32.bin   — for n in [4, 8, 16, ..., 1024]

Each .bin file contains (all little-endian):
  [4 bytes] int32_t  n            — transform length
  [8*n bytes] float32[2*n]        — interleaved input:  [r0,i0,...,rN-1,iN-1]
  [8*n bytes] float32[2*n]        — interleaved output: NumPy fft result

NumPy computes in float64 internally; results are converted to float32 and
saved.  The correctness test verifies that rvvfft output matches within
the allowed ULP error budget (≤ 4 ULP for f32, ≤ 4 ULP for f64).

Usage:
    python tests/gen_golden.py
    # → writes tests/golden/fft_*.bin
"""

from __future__ import annotations
import os
import struct
import sys
import math

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: NumPy is required.  Install with:  pip install numpy")

# ── Configuration ──────────────────────────────────────────────────────────

SIZES  = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
OUTDIR = os.path.join(os.path.dirname(__file__), "golden")

# Fixed random seed for reproducibility
RNG = np.random.default_rng(seed=0xDEADBEEF)


# ── Helpers ────────────────────────────────────────────────────────────────

def to_interleaved_f32(x: np.ndarray) -> bytes:
    """Convert complex128 array to interleaved float32 bytes."""
    arr = np.empty(2 * len(x), dtype=np.float32)
    arr[0::2] = x.real.astype(np.float32)
    arr[1::2] = x.imag.astype(np.float32)
    return arr.tobytes()


def write_golden(n: int, outdir: str) -> None:
    """Generate and write golden data for a length-n FFT."""
    # Random complex input
    inp = (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)).astype(np.complex128)

    # Convert to f32 first (our implementation operates in f32)
    inp_f32 = inp.astype(np.complex64)

    # Reference FFT: computed in f64 but from the exact f32 input values.
    # This gives the "correct" f64 result that our f32 implementation should
    # approximate within the allowed ULP budget.
    ref = np.fft.fft(inp_f32.astype(np.complex128))

    # Convert reference to f32
    ref_f32 = ref.astype(np.complex64)

    path = os.path.join(outdir, f"fft_{n}_f32.bin")
    with open(path, "wb") as f:
        # Header: n as int32
        f.write(struct.pack("<i", n))
        # Input samples (interleaved f32)
        f.write(to_interleaved_f32(inp_f32))
        # Reference output (interleaved f32)
        f.write(to_interleaved_f32(ref_f32))

    print(f"  wrote {path}  ({n} samples)")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Generating golden FFT data in {OUTDIR}/")
    for n in SIZES:
        write_golden(n, OUTDIR)
    print(f"\nDone: {len(SIZES)} files.")


if __name__ == "__main__":
    main()
