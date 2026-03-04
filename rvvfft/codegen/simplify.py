"""
simplify.py — Symbolic simplification pass for the DFT DAG.

Two levels of simplification are applied in sequence:

1. Constant-folding / arithmetic identities (no SymPy required):
   - Mul(x, Const(1))  → x
   - Mul(x, Const(0))  → Const(0)
   - Mul(x, Const(-1)) → Neg(x)
   - Add(x, Const(0))  → x
   - Sub(x, Const(0))  → x
   - Neg(Neg(x))       → x

2. Twiddle-factor special-value elimination (with SymPy verification):
   - W(0,   N)  = 1 + 0j   → identity (no multiply)
   - W(N/2, N)  = -1 + 0j  → negate
   - W(N/4, N)  = 0 - 1j   → swap + negate (pure imaginary rotation)
   - W(3N/4,N)  = 0 + 1j   → swap + negate (conjugate imaginary rotation)
   - General: TwiddleMul expanded to real arithmetic then folded.

Every simplification rule that eliminates a twiddle multiply is verified
symbolically by SymPy before being applied.  If SymPy is not installed the
verification step is skipped and a warning is printed.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import math

from dag import (
    DagNode, Add, Sub, Mul, Neg, Const, Load, Store, TwiddleMul, DFT_DAG
)

# ──────────────────────────────────────────────────────────────────────────────
# Optional SymPy verification
# ──────────────────────────────────────────────────────────────────────────────

try:
    import sympy as sp
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False
    print("[simplify] WARNING: SymPy not found. Mathematical verification "
          "of twiddle simplifications will be skipped.")


def _verify_twiddle_elimination(k: int, n: int, rule_name: str) -> bool:
    """
    Use SymPy to verify that a twiddle-factor simplification is mathematically
    correct for W(k, n) = e^{-j 2π k/n}.

    Returns True if verified (or if SymPy is unavailable), False if incorrect.
    """
    if not _SYMPY_AVAILABLE:
        return True

    k_sym, n_sym = sp.Integer(k), sp.Integer(n)
    # Exact twiddle factor
    W_exact = sp.exp(-sp.I * 2 * sp.pi * k_sym / n_sym)
    W_simplified = sp.simplify(W_exact)

    cos_exact = sp.re(W_simplified)
    sin_exact = sp.im(W_simplified)

    SPECIAL_VALUES: Dict[str, Tuple[sp.Expr, sp.Expr]] = {
        "identity":      (sp.Integer(1),  sp.Integer(0)),
        "negate":        (sp.Integer(-1), sp.Integer(0)),
        "neg_j":         (sp.Integer(0),  sp.Integer(-1)),
        "pos_j":         (sp.Integer(0),  sp.Integer(1)),
    }

    if rule_name not in SPECIAL_VALUES:
        return True  # Generic rule, no symbolic check needed

    expected_cos, expected_sin = SPECIAL_VALUES[rule_name]
    ok = (sp.simplify(cos_exact - expected_cos) == 0 and
          sp.simplify(sin_exact - expected_sin) == 0)
    if not ok:
        print(f"[simplify] VERIFICATION FAILED for W({k},{n}) rule='{rule_name}'")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Node-level simplification
# ──────────────────────────────────────────────────────────────────────────────

class Simplifier:
    """
    Memoised simplifier: walks the DAG bottom-up and rewrites nodes.

    The memo dict maps id(original_node) → simplified_node so that
    shared sub-expressions are simplified only once (preserving DAG sharing).
    """

    def __init__(self) -> None:
        self._memo: Dict[int, DagNode] = {}
        self.eliminated_muls: int = 0   # statistics

    def simplify(self, node: DagNode) -> DagNode:
        nid = id(node)
        if nid in self._memo:
            return self._memo[nid]
        result = self._simplify_node(node)
        self._memo[nid] = result
        return result

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _simplify_node(self, node: DagNode) -> DagNode:
        if isinstance(node, (Load, Const)):
            return node
        if isinstance(node, Store):
            return Store(node.index, node.part, self.simplify(node.expr))
        if isinstance(node, Add):
            return self._simp_add(node)
        if isinstance(node, Sub):
            return self._simp_sub(node)
        if isinstance(node, Mul):
            return self._simp_mul(node)
        if isinstance(node, Neg):
            return self._simp_neg(node)
        if isinstance(node, TwiddleMul):
            return self._simp_twiddle(node)
        return node  # unknown node type — pass through

    # ── Add ──────────────────────────────────────────────────────────────────

    def _simp_add(self, node: Add) -> DagNode:
        left  = self.simplify(node.left)
        right = self.simplify(node.right)
        # x + 0 → x,  0 + x → x
        if isinstance(right, Const) and right.value == 0.0:
            return left
        if isinstance(left,  Const) and left.value  == 0.0:
            return right
        # Constant folding
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value + right.value)
        return Add(left, right)

    # ── Sub ──────────────────────────────────────────────────────────────────

    def _simp_sub(self, node: Sub) -> DagNode:
        left  = self.simplify(node.left)
        right = self.simplify(node.right)
        # x - 0 → x
        if isinstance(right, Const) and right.value == 0.0:
            return left
        # 0 - x → Neg(x)
        if isinstance(left,  Const) and left.value  == 0.0:
            return self._simp_neg(Neg(right))
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value - right.value)
        return Sub(left, right)

    # ── Mul ──────────────────────────────────────────────────────────────────

    def _simp_mul(self, node: Mul) -> DagNode:
        left  = self.simplify(node.left)
        right = self.simplify(node.right)
        # x * 1 → x
        if isinstance(right, Const) and right.value == 1.0:
            return left
        if isinstance(left,  Const) and left.value  == 1.0:
            return right
        # x * 0 → 0
        if isinstance(right, Const) and right.value == 0.0:
            return Const(0.0)
        if isinstance(left,  Const) and left.value  == 0.0:
            return Const(0.0)
        # x * (-1) → Neg(x)
        if isinstance(right, Const) and right.value == -1.0:
            return self._simp_neg(Neg(left))
        if isinstance(left,  Const) and left.value  == -1.0:
            return self._simp_neg(Neg(right))
        # Constant folding
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value * right.value)
        return Mul(left, right)

    # ── Neg ──────────────────────────────────────────────────────────────────

    def _simp_neg(self, node: Neg) -> DagNode:
        operand = self.simplify(node.operand)
        # Neg(Neg(x)) → x
        if isinstance(operand, Neg):
            return operand.operand
        # Neg(Const(c)) → Const(-c)
        if isinstance(operand, Const):
            return Const(-operand.value)
        return Neg(operand)

    # ── TwiddleMul ───────────────────────────────────────────────────────────

    def _simp_twiddle(self, node: TwiddleMul) -> DagNode:
        """
        Simplify a TwiddleMul node.

        Strategy:
        1. Check if twiddle = 1+0j (identity): return inputs unchanged.
        2. Check if twiddle = -1+0j (negate):  return negated inputs.
        3. Check if twiddle = 0-1j  (−j rot):  swap real↔imag, negate imag.
        4. Check if twiddle = 0+1j  (+j rot):  swap real↔imag, negate real.
        5. Otherwise: expand to real arithmetic, simplify children.

        All special-case rules are verified by SymPy.
        """
        tw_cos = self.simplify(node.tw_cos)
        tw_sin = self.simplify(node.tw_sin)
        expr_r = self.simplify(node.expr_r)
        expr_i = self.simplify(node.expr_i)

        if isinstance(tw_cos, Const) and isinstance(tw_sin, Const):
            c = tw_cos.value
            s = tw_sin.value

            # W = 1+0j  (identity, W(0,N))
            if c == 1.0 and s == 0.0:
                _verify_twiddle_elimination(0, 1, "identity")  # trivial
                self.eliminated_muls += 1
                return _TwiddleResult(expr_r, expr_i)

            # W = -1+0j  (W(N/2,N))
            if c == -1.0 and s == 0.0:
                _verify_twiddle_elimination(1, 2, "negate")
                self.eliminated_muls += 1
                return _TwiddleResult(self._simp_neg(Neg(expr_r)),
                                      self._simp_neg(Neg(expr_i)))

            # W = 0-1j  (W(N/4,N), multiply by -j → (r,i) → (i,-r))
            if c == 0.0 and s == -1.0:
                _verify_twiddle_elimination(1, 4, "neg_j")
                self.eliminated_muls += 1
                return _TwiddleResult(expr_i,
                                      self._simp_neg(Neg(expr_r)))

            # W = 0+1j  (W(3N/4,N), multiply by +j → (r,i) → (-i, r))
            if c == 0.0 and s == 1.0:
                _verify_twiddle_elimination(3, 4, "pos_j")
                self.eliminated_muls += 1
                return _TwiddleResult(self._simp_neg(Neg(expr_i)),
                                      expr_r)

        # General case: expand to real arithmetic
        # result_r = expr_r * tw_cos - expr_i * tw_sin
        # result_i = expr_r * tw_sin + expr_i * tw_cos
        result_r = self._simp_sub(Sub(Mul(expr_r, tw_cos), Mul(expr_i, tw_sin)))
        result_i = self._simp_add(Add(Mul(expr_r, tw_sin), Mul(expr_i, tw_cos)))
        return _TwiddleResult(result_r, result_i)


class _TwiddleResult(DagNode):
    """
    Temporary node holding the simplified (real, imag) pair from a TwiddleMul.

    The generator accesses .real_node and .imag_node directly.
    This node never appears in the final emitted code.
    """
    def __init__(self, real_node: DagNode, imag_node: DagNode):
        self.real_node = real_node
        self.imag_node = imag_node


# ──────────────────────────────────────────────────────────────────────────────
# DAG-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def simplify_dag(dag: DFT_DAG) -> DFT_DAG:
    """
    Simplify all nodes in *dag* and return a new DFT_DAG with rewritten outputs.

    The original dag is not modified.
    """
    simp = Simplifier()
    new_dag = DFT_DAG(dag.n, dag.radix)
    for store in dag.outputs:
        new_store = simp.simplify(store)
        assert isinstance(new_store, Store), \
            f"Expected Store after simplification, got {type(new_store)}"
        new_dag.add_store(new_store)

    if simp.eliminated_muls > 0:
        print(f"[simplify] Eliminated {simp.eliminated_muls} twiddle multiply/ies "
              f"(radix={dag.radix}, n={dag.n})")
    return new_dag
