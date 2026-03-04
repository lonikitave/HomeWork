"""
rvvfft codelet generator — entry point.

Usage
-----
    python codegen/generator.py --verify

    python codegen/generator.py \\
        --radix 2 4 8 \\
        --layout interleaved split \\
        --dtype f32 f64 \\
        --vlen 256 1024 \\
        --outdir src/kernel/generated

The ``--verify`` flag runs the SymPy math-verification suite before
generating any code and aborts on the first failing rule.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from dag import build_dft_dag
from simplify import apply_simplifications
from verify_math import verify_all

# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rvvfft codelet generator")
    p.add_argument("--radix",  type=int, nargs="+", default=[2, 4, 8],
                   choices=[2, 4, 8, 16, 32])
    p.add_argument("--layout", nargs="+", default=["interleaved", "split"],
                   choices=["interleaved", "split"])
    p.add_argument("--dtype",  nargs="+", default=["f32", "f64"],
                   choices=["f32", "f64"])
    p.add_argument("--vlen",   type=int, nargs="+", default=[256, 1024],
                   choices=[256, 1024])
    p.add_argument("--outdir", default="src/kernel/generated",
                   help="Output directory for generated .c files")
    p.add_argument("--verify", action="store_true",
                   help="Run SymPy math-verification suite and exit")
    return p.parse_args()


# ── Code generation ────────────────────────────────────────────────────────

_CODEGEN_DIR = Path(__file__).parent
_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(_CODEGEN_DIR / "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
)


def generate_kernel(radix: int, layout: str, dtype: str, vlen: int,
                    outdir: Path) -> Path:
    """Generate one butterfly kernel and write it to *outdir*."""
    dag = build_dft_dag(radix)
    dag = apply_simplifications(dag, dtype=dtype)

    template = _TEMPLATE_ENV.get_template("butterfly.c.j2")
    code = template.render(
        radix=radix,
        layout=layout,
        dtype=dtype,
        vlen=vlen,
        dag=dag,
    )

    sew = 32 if dtype == "f32" else 64
    fname = f"bf_r{radix}_{layout}_{dtype}_vl{vlen}.c"
    out_path = outdir / fname
    out_path.write_text(code)
    print(f"  [generated] {out_path}")
    return out_path


def generate_registry(generated: list[Path], outdir: Path) -> None:
    """Write a C header that declares all generated codelets."""
    lines = ["/* Auto-generated codelet registry — do not edit manually */",
             "#pragma once", ""]
    for p in generated:
        name = p.stem
        lines.append(f"void {name}(const void *in, void *out, "
                     "const float *twiddle, int n);")
    (outdir / "codelet_registry.h").write_text("\n".join(lines) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("Running SymPy math-verification suite …")
    ok = verify_all()
    if not ok:
        print("ERROR: math verification failed — aborting code generation.",
              file=sys.stderr)
        sys.exit(1)
    print("All math rules verified ✓")
    if args.verify:
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        args.radix, args.layout, args.dtype, args.vlen
    ))
    print(f"Generating {len(combos)} kernel(s) → {outdir}")

    generated: list[Path] = []
    for radix, layout, dtype, vlen in combos:
        generated.append(generate_kernel(radix, layout, dtype, vlen, outdir))

    generate_registry(generated, outdir)
    print(f"Done. {len(generated)} kernel(s) written.")


if __name__ == "__main__":
    main()
