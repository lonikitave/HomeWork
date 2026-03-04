"""
dag.py — DFT Computation DAG (Directed Acyclic Graph)

Represents a DFT butterfly computation as a graph of symbolic nodes.
Each node is one of:
  - Load(index, part)           : load real/imag component from input
  - Store(index, part, expr)    : write result to output
  - Const(value)                : a numeric literal (twiddle factor component)
  - Add(left, right)            : floating-point addition
  - Sub(left, right)            : floating-point subtraction
  - Mul(left, right)            : floating-point multiplication
  - Neg(operand)                : negation (unary minus)
  - TwiddleMul(expr_r, expr_i,  : multiply by twiddle factor W(k,N)
               tw_cos, tw_sin)

This representation is backend-agnostic; the generator layer maps nodes
to concrete RVV intrinsic calls.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union
import math


# ──────────────────────────────────────────────────────────────────────────────
# Node base
# ──────────────────────────────────────────────────────────────────────────────

class DagNode:
    """Abstract base for all DAG nodes."""

    def __add__(self, other: "DagNode") -> "Add":
        return Add(self, _wrap(other))

    def __radd__(self, other: "DagNode") -> "Add":
        return Add(_wrap(other), self)

    def __sub__(self, other: "DagNode") -> "Sub":
        return Sub(self, _wrap(other))

    def __rsub__(self, other: "DagNode") -> "Sub":
        return Sub(_wrap(other), self)

    def __mul__(self, other: "DagNode") -> "Mul":
        return Mul(self, _wrap(other))

    def __rmul__(self, other: "DagNode") -> "Mul":
        return Mul(_wrap(other), self)

    def __neg__(self) -> "Neg":
        return Neg(self)

    def inputs(self) -> List["DagNode"]:
        """Return all child nodes (for graph traversal)."""
        return []

    def is_const(self) -> bool:
        return False

    def const_value(self) -> float:
        raise TypeError("Not a constant node")


def _wrap(x) -> DagNode:
    """Coerce a Python number into a Const node."""
    if isinstance(x, DagNode):
        return x
    return Const(float(x))


# ──────────────────────────────────────────────────────────────────────────────
# Leaf nodes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Load(DagNode):
    """Load the real or imag part of input sample at position *index*."""
    index: int
    part: str  # 'r' or 'i'

    def __repr__(self) -> str:
        return f"Load({self.index},{self.part})"


@dataclass(eq=False)
class Const(DagNode):
    """A compile-time constant (used for twiddle factor components)."""
    value: float

    def is_const(self) -> bool:
        return True

    def const_value(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value:.6g})"


# ──────────────────────────────────────────────────────────────────────────────
# Arithmetic nodes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Add(DagNode):
    left: DagNode
    right: DagNode

    def inputs(self):
        return [self.left, self.right]

    def __repr__(self) -> str:
        return f"Add({self.left!r}, {self.right!r})"


@dataclass(eq=False)
class Sub(DagNode):
    left: DagNode
    right: DagNode

    def inputs(self):
        return [self.left, self.right]

    def __repr__(self) -> str:
        return f"Sub({self.left!r}, {self.right!r})"


@dataclass(eq=False)
class Mul(DagNode):
    left: DagNode
    right: DagNode

    def inputs(self):
        return [self.left, self.right]

    def __repr__(self) -> str:
        return f"Mul({self.left!r}, {self.right!r})"


@dataclass(eq=False)
class Neg(DagNode):
    operand: DagNode

    def inputs(self):
        return [self.operand]

    def __repr__(self) -> str:
        return f"Neg({self.operand!r})"


@dataclass(eq=False)
class TwiddleMul(DagNode):
    """
    Complex multiply of (expr_r + j*expr_i) by twiddle factor (tw_cos + j*tw_sin).

    Result real part = expr_r * tw_cos - expr_i * tw_sin
    Result imag part = expr_r * tw_sin + expr_i * tw_cos

    When tw_cos and tw_sin are Const nodes with special values (±1, ±0),
    the simplify pass eliminates the multiplication entirely.
    """
    expr_r: DagNode   # real part of input expression
    expr_i: DagNode   # imag part of input expression
    tw_cos: DagNode   # cos(2π k/N) — twiddle real component
    tw_sin: DagNode   # sin(2π k/N) — twiddle imag component  (note: −sin for W^k)

    def inputs(self):
        return [self.expr_r, self.expr_i, self.tw_cos, self.tw_sin]

    def real(self) -> DagNode:
        """Expand to real-part expression (before simplification)."""
        return Sub(Mul(self.expr_r, self.tw_cos),
                   Mul(self.expr_i, self.tw_sin))

    def imag(self) -> DagNode:
        """Expand to imag-part expression (before simplification)."""
        return Add(Mul(self.expr_r, self.tw_sin),
                   Mul(self.expr_i, self.tw_cos))

    def __repr__(self) -> str:
        return (f"TwiddleMul(expr=({self.expr_r!r}, {self.expr_i!r}), "
                f"twiddle=({self.tw_cos!r}, {self.tw_sin!r}))")


# ──────────────────────────────────────────────────────────────────────────────
# Output node
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Store(DagNode):
    """Write *expr* to output sample at position *index*, real or imag part."""
    index: int
    part: str      # 'r' or 'i'
    expr: DagNode

    def inputs(self):
        return [self.expr]

    def __repr__(self) -> str:
        return f"Store({self.index},{self.part},{self.expr!r})"


# ──────────────────────────────────────────────────────────────────────────────
# DAG container
# ──────────────────────────────────────────────────────────────────────────────

class DFT_DAG:
    """
    Full DAG representing an N-point DFT butterfly stage.

    Attributes:
        n       : transform length
        radix   : radix of this stage (2, 4, 8, …)
        outputs : list of Store nodes (one per output sample × 2 parts)
    """

    def __init__(self, n: int, radix: int):
        self.n = n
        self.radix = radix
        self.outputs: List[Store] = []

    def add_store(self, store: Store) -> None:
        self.outputs.append(store)

    def all_nodes(self) -> List[DagNode]:
        """Topological traversal of all nodes reachable from outputs."""
        visited = set()
        order: List[DagNode] = []

        def visit(node: DagNode) -> None:
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            for child in node.inputs():
                visit(child)
            order.append(node)

        for s in self.outputs:
            visit(s)
        return order

    def __repr__(self) -> str:
        return f"DFT_DAG(n={self.n}, radix={self.radix}, stores={len(self.outputs)})"


# ──────────────────────────────────────────────────────────────────────────────
# DAG builders
# ──────────────────────────────────────────────────────────────────────────────

def twiddle_const(k: int, n: int) -> tuple[DagNode, DagNode]:
    """
    Return (cos_node, sin_node) for the twiddle factor W(k, n) = e^{-j 2π k/n}.

    Special values are returned as exact Const nodes so the simplify pass
    can eliminate the multiplication.
    """
    angle = -2.0 * math.pi * k / n
    cos_v = math.cos(angle)
    sin_v = math.sin(angle)

    # Round near-integer values to exact integers for clean simplification
    def snap(v: float) -> float:
        if abs(v - round(v)) < 1e-12:
            return float(round(v))
        return v

    return Const(snap(cos_v)), Const(snap(sin_v))


def build_radix2_butterfly(group: int, n: int) -> DFT_DAG:
    """
    Build a single radix-2 Cooley-Tukey butterfly DAG for group index *group*.

    Computes:
        out[0] = in[0] + W(group, n) * in[1]
        out[1] = in[0] - W(group, n) * in[1]

    where in[0] and in[1] are the two input elements.
    """
    dag = DFT_DAG(n=2, radix=2)

    r0 = Load(0, 'r')
    i0 = Load(0, 'i')
    r1 = Load(1, 'r')
    i1 = Load(1, 'i')

    cos_w, sin_w = twiddle_const(group, n)
    tw = TwiddleMul(r1, i1, cos_w, sin_w)

    tw_r = tw.real()
    tw_i = tw.imag()

    dag.add_store(Store(0, 'r', r0 + tw_r))
    dag.add_store(Store(0, 'i', i0 + tw_i))
    dag.add_store(Store(1, 'r', r0 - tw_r))
    dag.add_store(Store(1, 'i', i0 - tw_i))

    return dag


def build_radix4_butterfly(group: int, n: int) -> DFT_DAG:
    """
    Build a radix-4 DIT butterfly DAG.

    Decomposes a 4-point DFT into additions/subtractions and twiddle multiplies,
    exploiting that W(n/4, n) = -j  (imag-unit rotation).
    """
    dag = DFT_DAG(n=4, radix=4)

    # Load 4 complex inputs
    x = [(Load(k, 'r'), Load(k, 'i')) for k in range(4)]

    # Twiddle factors: W^0=1, W^k for k=group, 2*group, 3*group
    twiddles = [twiddle_const(group * k, n) for k in range(4)]

    # Apply twiddles
    xt: list[tuple[DagNode, DagNode]] = []
    for k in range(4):
        if k == 0:
            xt.append(x[0])  # W^0 = 1, no multiply
        else:
            tw = TwiddleMul(x[k][0], x[k][1], twiddles[k][0], twiddles[k][1])
            xt.append((tw.real(), tw.imag()))

    # Stage 1: two radix-2 butterflies
    a0r = xt[0][0] + xt[2][0]
    a0i = xt[0][1] + xt[2][1]
    a1r = xt[0][0] - xt[2][0]
    a1i = xt[0][1] - xt[2][1]

    a2r = xt[1][0] + xt[3][0]
    a2i = xt[1][1] + xt[3][1]
    # Multiply (xt[1] - xt[3]) by -j: (r,i) -> (i, -r)
    a3r_pre = xt[1][0] - xt[3][0]
    a3i_pre = xt[1][1] - xt[3][1]
    a3r = a3i_pre          # multiply by -j: real part becomes imag
    a3i = Neg(a3r_pre)     # imag part becomes -real

    # Stage 2: combine
    dag.add_store(Store(0, 'r', a0r + a2r))
    dag.add_store(Store(0, 'i', a0i + a2i))
    dag.add_store(Store(1, 'r', a1r + a3r))
    dag.add_store(Store(1, 'i', a1i + a3i))
    dag.add_store(Store(2, 'r', a0r - a2r))
    dag.add_store(Store(2, 'i', a0i - a2i))
    dag.add_store(Store(3, 'r', a1r - a3r))
    dag.add_store(Store(3, 'i', a1i - a3i))

    return dag


def build_radix8_butterfly(group: int, n: int) -> DFT_DAG:
    """
    Build a radix-8 DIT butterfly DAG.

    Uses the split-radix / Winograd structure:
      decompose 8-point DFT into two radix-2 and one radix-4 sub-problem,
      exploiting special twiddle values (±1, ±j, ±(1±j)/√2).
    """
    dag = DFT_DAG(n=8, radix=8)

    # Load 8 complex inputs
    x = [(Load(k, 'r'), Load(k, 'i')) for k in range(8)]

    # Twiddle factors W^(group*k) for k=0..7
    twiddles = [twiddle_const(group * k, n) for k in range(8)]

    # Apply twiddles (skip W^0 = 1)
    xt: list[tuple[DagNode, DagNode]] = []
    for k in range(8):
        if k == 0:
            xt.append(x[0])
        else:
            tw = TwiddleMul(x[k][0], x[k][1], twiddles[k][0], twiddles[k][1])
            xt.append((tw.real(), tw.imag()))

    # Stage 1: 4 radix-2 butterflies across stride-4 pairs
    def bf2(a: tuple[DagNode, DagNode],
            b: tuple[DagNode, DagNode]) -> tuple[tuple, tuple]:
        return (a[0] + b[0], a[1] + b[1]), (a[0] - b[0], a[1] - b[1])

    s0, s4 = bf2(xt[0], xt[4])
    s1, s5 = bf2(xt[1], xt[5])
    s2, s6 = bf2(xt[2], xt[6])
    s3, s7 = bf2(xt[3], xt[7])

    # Stage 2: 2 radix-4 sub-problems
    # Sub-problem A (even indices: s0,s1,s2,s3) — twiddles W^0,W^2,W^4,W^6
    # Sub-problem B (odd indices:  s4,s5,s6,s7) — twiddles W^0,W^1,W^2,W^3

    # --- Sub-problem A ---
    t0, t2 = bf2(s0, s2)
    t1, t3 = bf2(s1, s3)
    # Multiply t3 by -j
    t3r = t3[1]
    t3i = Neg(t3[0])

    out0 = (t0[0] + t1[0], t0[1] + t1[1])
    out2 = (t0[0] - t1[0], t0[1] - t1[1])
    out4 = (t2[0] + t3r,   t2[1] + t3i)
    out6 = (t2[0] - t3r,   t2[1] - t3i)

    # --- Sub-problem B: apply additional twiddle W^1 to s5, W^2 to s6, W^3 to s7 ---
    W1 = twiddle_const(group, n)
    W2 = twiddle_const(group * 2, n)
    W3 = twiddle_const(group * 3, n)

    def tw_mul(pr: DagNode, pi: DagNode,
               cr: DagNode, ci: DagNode) -> tuple[DagNode, DagNode]:
        t = TwiddleMul(pr, pi, cr, ci)
        return t.real(), t.imag()

    s5t = tw_mul(s5[0], s5[1], W1[0], W1[1])
    s6t = tw_mul(s6[0], s6[1], W2[0], W2[1])
    s7t = tw_mul(s7[0], s7[1], W3[0], W3[1])

    u0, u2 = bf2(s4, s6t)
    u1, u3 = bf2(s5t, s7t)
    u3r = u3[1]
    u3i = Neg(u3[0])

    out1 = (u0[0] + u1[0], u0[1] + u1[1])
    out3 = (u0[0] - u1[0], u0[1] - u1[1])
    out5 = (u2[0] + u3r,   u2[1] + u3i)
    out7 = (u2[0] - u3r,   u2[1] - u3i)

    for idx, (outr, outi) in enumerate(
            [out0, out1, out2, out3, out4, out5, out6, out7]):
        dag.add_store(Store(idx, 'r', outr))
        dag.add_store(Store(idx, 'i', outi))

    return dag
