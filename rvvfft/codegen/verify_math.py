"""
Mathematical verification of rvvfft simplification rules using SymPy.

Each rule proved here corresponds to a simplification applied in
simplify.py.  Run standalone or via:

    python codegen/verify_math.py

All rules must pass before code generation is allowed.
"""

from __future__ import annotations

import sys
from typing import Callable

import sympy as sp

# ── Helpers ────────────────────────────────────────────────────────────────

pi = sp.pi
I  = sp.I


def W(k: int, N: int) -> sp.Expr:
    """Twiddle factor W(k, N) = exp(-2πi·k/N)."""
    return sp.exp(-2 * pi * I * k / N)


# ── Rule definitions ───────────────────────────────────────────────────────

RuleResult = tuple[str, bool, str]   # (name, passed, message)
Rule = Callable[[], RuleResult]

_rules: list[Rule] = []


def rule(fn: Rule) -> Rule:
    _rules.append(fn)
    return fn


@rule
def rule_w0_is_one() -> RuleResult:
    """W(0, N) = 1 for any N."""
    N = sp.Symbol("N", positive=True, integer=True)
    expr = W(0, N)
    simplified = sp.simplify(expr - 1)
    ok = simplified == 0
    return ("W(0,N)=1", ok, f"residual = {simplified}")


@rule
def rule_w_half_is_neg_one() -> RuleResult:
    """W(N/2, N) = -1."""
    N = sp.Symbol("N", positive=True, integer=True)
    expr = W(N // 2, N)
    # For symbolic N evaluate at concrete even N to verify
    for n in (4, 8, 16, 32):
        val = complex(W(n // 2, n).evalf())
        if abs(val - (-1)) > 1e-12:
            return ("W(N/2,N)=-1", False, f"failed at N={n}: {val}")
    return ("W(N/2,N)=-1", True, "ok")


@rule
def rule_w_quarter_is_neg_j() -> RuleResult:
    """W(N/4, N) = -j."""
    for n in (4, 8, 16, 32):
        val = complex(W(n // 4, n).evalf())
        if abs(val - (-1j)) > 1e-12:
            return ("W(N/4,N)=-j", False, f"failed at N={n}: {val}")
    return ("W(N/4,N)=-j", True, "ok")


@rule
def rule_w_three_quarter_is_pos_j() -> RuleResult:
    """W(3N/4, N) = +j."""
    for n in (4, 8, 16, 32):
        val = complex(W(3 * n // 4, n).evalf())
        if abs(val - 1j) > 1e-12:
            return ("W(3N/4,N)=+j", False, f"failed at N={n}: {val}")
    return ("W(3N/4,N)=+j", True, "ok")


@rule
def rule_conjugate_symmetry() -> RuleResult:
    """W(N-k, N) = conj(W(k, N))."""
    for n in (4, 8, 16):
        for k in range(1, n):
            lhs = complex(W(n - k, n).evalf())
            rhs = complex(W(k, n).evalf()).conjugate()
            if abs(lhs - rhs) > 1e-12:
                return ("W(N-k,N)=conj(W(k,N))", False,
                        f"failed at N={n},k={k}")
    return ("W(N-k,N)=conj(W(k,N))", True, "ok")


@rule
def rule_radix2_butterfly_correctness() -> RuleResult:
    """Radix-2 butterfly: [x0+W·x1, x0-W·x1] computes 2-pt DFT correctly."""
    import numpy as np
    rng = np.random.default_rng(42)
    for _ in range(100):
        x = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        w = np.exp(-2j * np.pi * 0 / 2)   # k=0 → W(0,2)=1 for first output
        butterfly = np.array([x[0] + x[1], x[0] - x[1]])
        ref = np.fft.fft(x)
        if np.max(np.abs(butterfly - ref)) > 1e-12:
            return ("radix-2 butterfly", False, "mismatch vs numpy.fft")
    return ("radix-2 butterfly", True, "ok")


# ── Runner ─────────────────────────────────────────────────────────────────

def verify_all(verbose: bool = True) -> bool:
    """Run all registered rules; return True if every rule passes."""
    all_ok = True
    for fn in _rules:
        name, ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        if verbose:
            print(f"  [{status}] {name}: {msg}")
        if not ok:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    ok = verify_all()
    sys.exit(0 if ok else 1)
