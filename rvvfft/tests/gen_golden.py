"""
Generate golden reference data for correctness testing.

Computes FFT results using NumPy (double precision) and writes them as
binary files under tests/golden/.  The C test driver (test_correctness.c)
reads these files and compares against the rvvfft output.

Usage
-----
    python tests/gen_golden.py [--outdir tests/golden] [--sizes 64 128 256 512 1024]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def gen_golden(n: int, outdir: Path) -> None:
    """Generate and write golden data for an n-point FFT."""
    rng = np.random.default_rng(seed=n)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x = x.astype(np.complex64)               # single precision input

    y_ref = np.fft.fft(x.astype(np.complex128)).astype(np.complex64)

    # Binary format: [N (int32)] [x_re f32 × N] [x_im f32 × N]
    #                           [y_re f32 × N] [y_im f32 × N]
    fname = outdir / f"fft_{n}.bin"
    with fname.open("wb") as f:
        f.write(struct.pack("<i", n))
        f.write(x.real.astype("<f4").tobytes())
        f.write(x.imag.astype("<f4").tobytes())
        f.write(y_ref.real.astype("<f4").tobytes())
        f.write(y_ref.imag.astype("<f4").tobytes())
    print(f"  [golden] {fname}  ({n}-pt)")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate rvvfft golden data")
    p.add_argument("--outdir", default="tests/golden")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64, 128, 256, 512, 1024,
                            2048, 4096, 8192, 65536, 1048576])
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for n in args.sizes:
        gen_golden(n, outdir)

    print(f"Generated {len(args.sizes)} golden file(s) in {outdir}/")


if __name__ == "__main__":
    main()
