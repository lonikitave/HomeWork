"""
DFT computation DAG (Directed Acyclic Graph) builder.

Each node represents one arithmetic operation over complex numbers.
The DAG is used as the intermediate representation fed into the
simplification pass (simplify.py) and the template engine (Jinja2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ── Node types ─────────────────────────────────────────────────────────────

@dataclass
class Node:
    op: str          # 'load' | 'store' | 'add' | 'sub' | 'mul' | 'twiddle'
    inputs: List["Node"] = field(default_factory=list)
    # For twiddle nodes: W(k, N) = exp(-2πi·k/N)
    twiddle_k: Optional[int] = None
    twiddle_n: Optional[int] = None
    # Index into the input/output array (load/store nodes)
    index: Optional[int] = None
    # Human-readable label for debugging
    label: str = ""


def _load(idx: int) -> Node:
    return Node(op="load", index=idx, label=f"x[{idx}]")


def _twiddle_mul(node: Node, k: int, n: int) -> Node:
    return Node(op="twiddle", inputs=[node], twiddle_k=k, twiddle_n=n,
                label=f"W({k},{n})*{node.label}")


def _add(a: Node, b: Node) -> Node:
    return Node(op="add", inputs=[a, b], label=f"({a.label}+{b.label})")


def _sub(a: Node, b: Node) -> Node:
    return Node(op="sub", inputs=[a, b], label=f"({a.label}-{b.label})")


def _store(idx: int, src: Node) -> Node:
    return Node(op="store", index=idx, inputs=[src],
                label=f"y[{idx}]={src.label}")


# ── Radix-N butterfly builders ─────────────────────────────────────────────

def _radix2_butterfly(inputs: List[Node], base_k: int, n: int
                      ) -> List[Node]:
    """One Cooley-Tukey radix-2 DIT butterfly stage."""
    assert len(inputs) == 2
    x0, x1 = inputs
    tw = _twiddle_mul(x1, base_k, n)
    return [_add(x0, tw), _sub(x0, tw)]


def _radix4_butterfly(inputs: List[Node], base_k: int, n: int
                      ) -> List[Node]:
    """One radix-4 DIT butterfly stage (reduces twiddle multiplications)."""
    assert len(inputs) == 4
    x0, x1, x2, x3 = inputs

    tw1 = _twiddle_mul(x1, base_k,     n)
    tw2 = _twiddle_mul(x2, 2 * base_k, n)
    tw3 = _twiddle_mul(x3, 3 * base_k, n)

    s0 = _add(x0,  tw2)
    s1 = _sub(x0,  tw2)
    s2 = _add(tw1, tw3)
    s3 = _sub(tw1, tw3)

    return [_add(s0, s2), _sub(s1, s3), _sub(s0, s2), _add(s1, s3)]


def _radix8_butterfly(inputs: List[Node], base_k: int, n: int
                      ) -> List[Node]:
    """One radix-8 DIT butterfly stage."""
    assert len(inputs) == 8
    # Implement via two radix-4 stages for clarity
    first  = _radix4_butterfly(inputs[:4], base_k,     n)
    second = _radix4_butterfly(inputs[4:], base_k * 2, n)
    # Combine
    out = []
    for i in range(4):
        tw = _twiddle_mul(second[i], base_k * i, n)
        out.append(_add(first[i], tw))
        out.append(_sub(first[i], tw))
    return out


_BUTTERFLY_BUILDERS = {
    2: _radix2_butterfly,
    4: _radix4_butterfly,
    8: _radix8_butterfly,
}


# ── Top-level DAG builder ──────────────────────────────────────────────────

def build_dft_dag(radix: int) -> dict:
    """
    Build the DFT computation DAG for a single butterfly of the given radix.

    Returns a dict with:
        inputs  – list of input Load nodes
        outputs – list of Store nodes
        radix   – the radix used
    """
    if radix not in _BUTTERFLY_BUILDERS:
        raise ValueError(f"Unsupported radix {radix}. "
                         f"Supported: {list(_BUTTERFLY_BUILDERS)}")

    n = radix  # one butterfly block of size `radix`
    loads = [_load(i) for i in range(n)]

    builder = _BUTTERFLY_BUILDERS[radix]
    results = builder(loads, base_k=0, n=n)

    stores = [_store(i, results[i]) for i in range(n)]

    return {"inputs": loads, "outputs": stores, "radix": radix}
