"""
Symbolic simplification pass for rvvfft DFT DAGs.

Rules applied
-------------
1. W(0, N) = 1          → identity (no multiplication needed)
2. W(N/4, N) = −j       → replace with negated imaginary swap
3. W(N/2, N) = −1       → replace with negation
4. W(3N/4, N) = +j      → replace with imaginary swap
5. W(k, N) = conj(W(N−k, N))  (not simplified, but normalised to k < N/2)

Every simplification rule is symbolically verified in verify_math.py using
SymPy before any code is emitted.
"""

from __future__ import annotations

from typing import Optional

from dag import Node


# ── Twiddle-factor special-value detection ─────────────────────────────────

def _twiddle_class(k: int, n: int) -> str:
    """Return a string tag for well-known twiddle values."""
    k = k % n
    if k == 0:
        return "one"          # W(0,N) = 1
    if 4 * k == n:
        return "neg_j"        # W(N/4,N) = -j
    if 2 * k == n:
        return "neg_one"      # W(N/2,N) = -1
    if 4 * k == 3 * n:
        return "pos_j"        # W(3N/4,N) = +j
    return "general"


# ── Per-node simplification ────────────────────────────────────────────────

def _simplify_node(node: Node) -> Node:
    if node.op != "twiddle":
        return node

    cls = _twiddle_class(node.twiddle_k, node.twiddle_n)

    if cls == "one":
        # W(0,N)*x = x  → drop the multiplication
        child = node.inputs[0]
        child.label = f"(1·{child.label}→{child.label})"
        return child

    if cls in ("neg_one", "neg_j", "pos_j"):
        # Mark the node with a simplified op so the template can emit
        # cheap sign/swap instructions instead of a full complex multiply.
        node.op = f"twiddle_{cls}"

    return node


# ── DAG-wide simplification pass ──────────────────────────────────────────

def _walk(node: Node, memo: dict) -> Node:
    nid = id(node)
    if nid in memo:
        return memo[nid]
    node.inputs = [_walk(child, memo) for child in node.inputs]
    result = _simplify_node(node)
    memo[nid] = result
    return result


def apply_simplifications(dag: dict, dtype: str = "f32") -> dict:
    """
    Apply twiddle-factor simplifications to every node in *dag* in-place.

    Parameters
    ----------
    dag:   dict returned by ``build_dft_dag``
    dtype: 'f32' or 'f64' (currently unused but reserved for future rules)
    """
    memo: dict = {}
    dag["outputs"] = [_walk(node, memo) for node in dag["outputs"]]
    return dag
