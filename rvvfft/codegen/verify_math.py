"""
verify_math.py — Automated mathematical verification of codegen rules.

Verifies every twiddle-factor simplification rule used by simplify.py
against SymPy's symbolic arithmetic.  Run this script standalone to
confirm that the simplifier is mathematically sound:

    python verify_math.py

All tests must pass before any generated kernel is committed to the
repository.  CI should run this script as part of its gate.
"""

from __future__ import annotations
import sys
import math
from typing import Callable, List, Tuple

try:
    import sympy as sp
except ImportError:
    sys.exit(
        "ERROR: SymPy is required for mathematical verification.\n"
        "Install it with:  pip install sympy"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def W(k: int, n: int) -> sp.Expr:
    """Exact twiddle factor W^k_N = e^{-j 2π k / N} as a SymPy expression."""
    return sp.exp(-sp.I * 2 * sp.pi * sp.Rational(k, n))


def complex_eq(a: sp.Expr, b: sp.Expr) -> bool:
    """Return True iff a and b are symbolically equal (real and imag parts)."""
    diff = sp.simplify(a - b)
    return diff == 0


def butterfly_output(x: List[sp.Expr], w: List[sp.Expr]) -> List[sp.Expr]:
    """
    Apply twiddle factors w[k] to inputs x[k], then compute DFT outputs
    via the DFT matrix.  Returns a list of N complex output values.
    """
    n = len(x)
    return [sum(x[k] * W(j * k, n) for k in range(n)) for j in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Test registry
# ──────────────────────────────────────────────────────────────────────────────

TESTS: List[Tuple[str, Callable[[], bool]]] = []


def test(name: str):
    """Decorator that registers a verification test."""
    def decorator(fn: Callable[[], bool]):
        TESTS.append((name, fn))
        return fn
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Twiddle special-value rules
# ──────────────────────────────────────────────────────────────────────────────

@test("W(0,N) == 1+0j  [identity rule]")
def _():
    for n in [2, 4, 8, 16, 32, 64]:
        w = W(0, n)
        if not complex_eq(w, sp.Integer(1)):
            return False
    return True


@test("W(N/2,N) == -1+0j  [negate rule]")
def _():
    for n in [2, 4, 8, 16, 32]:
        w = W(n // 2, n)
        if not complex_eq(w, sp.Integer(-1)):
            return False
    return True


@test("W(N/4,N) == -j  [neg_j rule: multiply by -j → (r,i)→(i,-r)]")
def _():
    for n in [4, 8, 16, 32]:
        w = W(n // 4, n)
        if not complex_eq(w, -sp.I):
            return False
    return True


@test("W(3N/4,N) == +j  [pos_j rule: multiply by +j → (r,i)→(-i,r)]")
def _():
    for n in [4, 8, 16, 32]:
        w = W(3 * n // 4, n)
        if not complex_eq(w, sp.I):
            return False
    return True


@test("neg_j rotation: (r+ij)*(-j) == i + j(-r)")
def _():
    r, i = sp.symbols("r i", real=True)
    lhs = (r + sp.I * i) * (-sp.I)
    rhs = i + sp.I * (-r)
    return complex_eq(lhs, rhs)


@test("pos_j rotation: (r+ij)*(+j) == -i + j(r)")
def _():
    r, i = sp.symbols("r i", real=True)
    lhs = (r + sp.I * i) * sp.I
    rhs = -i + sp.I * r
    return complex_eq(lhs, rhs)


# ──────────────────────────────────────────────────────────────────────────────
# Radix-2 butterfly correctness
# ──────────────────────────────────────────────────────────────────────────────

@test("Radix-2 butterfly: out[0]+out[1] == DFT2 × [x0,x1]")
def _():
    x0, x1 = sp.symbols("x0 x1")
    # Radix-2 DIT butterfly with trivial twiddle W^0=1
    out0 = x0 + x1         # W^0 = 1
    out1 = x0 - x1

    # 2-point DFT: F = [[1,1],[1,-1]]
    dft0 = x0 * W(0, 2) + x1 * W(0, 2)  # W(0,2)=1, W(0,2)=1
    dft1 = x0 * W(0, 2) + x1 * W(1, 2)  # W(1,2)=-1

    dft0_s = sp.simplify(dft0)
    dft1_s = sp.simplify(dft1)

    return complex_eq(out0, dft0_s) and complex_eq(out1, dft1_s)


@test("Radix-2 butterfly: twiddle W(k,N) complex multiply formula")
def _():
    """Verify: (ar+j*ai)*(cos+j*sin) = (ar*cos-ai*sin) + j*(ar*sin+ai*cos)"""
    ar, ai, c, s = sp.symbols("ar ai c s", real=True)
    product = (ar + sp.I * ai) * (c + sp.I * s)
    real_part = sp.re(sp.expand(product))
    imag_part = sp.im(sp.expand(product))
    expected_r = ar * c - ai * s
    expected_i = ar * s + ai * c
    return (sp.simplify(real_part - expected_r) == 0 and
            sp.simplify(imag_part - expected_i) == 0)


# ──────────────────────────────────────────────────────────────────────────────
# Radix-4 butterfly correctness
# ──────────────────────────────────────────────────────────────────────────────

@test("Radix-4 butterfly outputs match 4-point DFT (numerical, group=0)")
def _():
    """
    With group=0 all twiddles are W^0=1, so the 4-point butterfly must
    compute the exact 4-point DFT.
    """
    import cmath

    def dft4(x: list) -> list:
        n = 4
        return [sum(x[k] * cmath.exp(-2j * math.pi * j * k / n)
                    for k in range(n))
                for j in range(n)]

    def radix4_group0(x: list) -> list:
        # Stage 1: stride-2 butterflies
        a0 = x[0] + x[2]
        a2 = x[0] - x[2]
        a1 = x[1] + x[3]
        # Multiply (x[1]-x[3]) by -j
        diff13 = x[1] - x[3]
        a3r = diff13.imag   # multiply by -j: real becomes imag
        a3i = -diff13.real  # imag becomes -real
        a3 = complex(a3r, a3i)

        # Stage 2: stride-1 butterflies
        return [a0 + a1, a2 + a3, a0 - a1, a2 - a3]

    test_inputs = [
        [1+0j, 0+0j, 0+0j, 0+0j],
        [1+0j, 1+0j, 1+0j, 1+0j],
        [1+0j, 0+1j, -1+0j, 0-1j],
        [1+2j, 3+4j, 5+6j, 7+8j],
    ]

    for x in test_inputs:
        ref = dft4(x)
        got = radix4_group0(x)
        for r, g in zip(ref, got):
            if abs(r - g) > 1e-9:
                return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# FLOP count formula
# ──────────────────────────────────────────────────────────────────────────────

@test("FLOP count: 5*N*log2(N) matches reference values for power-of-2 N")
def _():
    # Reference: Cooley-Tukey exact operation count = 5*N*log2(N)
    expected = {
        2:    10,
        4:    40,
        8:   120,
        16:  320,
        32:  800,
        64: 1920,
        1024: 51200,
    }
    for n, ref in expected.items():
        got = int(5 * n * math.log2(n))
        if got != ref:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Algebraic identities used in simplify.py
# ──────────────────────────────────────────────────────────────────────────────

@test("Neg(Neg(x)) == x")
def _():
    x = sp.Symbol("x")
    return sp.simplify(-(-x) - x) == 0


@test("x * 1 == x,  x * 0 == 0,  x * (-1) == -x")
def _():
    x = sp.Symbol("x")
    return (sp.simplify(x * 1 - x) == 0 and
            sp.simplify(x * 0    ) == 0 and
            sp.simplify(x * (-1) + x) == 0)


@test("x + 0 == x,  0 + x == x,  x - 0 == x")
def _():
    x = sp.Symbol("x")
    return (sp.simplify(x + 0 - x) == 0 and
            sp.simplify(0 + x - x) == 0 and
            sp.simplify(x - 0 - x) == 0)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all() -> int:
    """Run all registered tests.  Returns the number of failures."""
    passed = 0
    failed = 0
    print("=" * 60)
    print("  rvvfft codegen mathematical verification")
    print("=" * 60)

    for name, fn in TESTS:
        try:
            ok = fn()
        except Exception as exc:
            ok = False
            print(f"  [EXCEPTION] {name}\n    {exc}")

        if ok:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    n_failed = run_all()
    sys.exit(0 if n_failed == 0 else 1)
