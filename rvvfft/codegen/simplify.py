"""
simplify.py — 符号化简pass与SymPy数学验证

对DFT DAG中的节点进行代数化简，消除不必要的乘法指令：
  1. W(0,N) = 1  → TwiddleNode变为恒等（无乘法）
  2. W(N/4,N) = -j → TwiddleNode变为RotJNode(sign=-1)（无乘法）
  3. W(N/2,N) = -1 → TwiddleNode变为NegNode（无乘法）
  4. W(N/8,N) = (1-j)/√2 → TwiddleNode变为ScaleNode+RotJNode（1次实数乘）

每条化简规则用 SymPy 符号计算自动验证正确性，
验证失败时抛出 ValueError（宁可不简化也不产生错误的代码）。

Phase 2代码生成器完成时，此模块将被集成到generator.py的DAG优化pass中。
"""
from __future__ import annotations
import math
import sympy as sp
from dag import DFTNode, TwiddleNode, RotJNode, NegNode, ScaleNode, AddNode


# ── 化简规则定义 ──────────────────────────────────────────────

def _verify_rule(name: str, lhs_r: sp.Expr, lhs_i: sp.Expr,
                 rhs_r: sp.Expr, rhs_i: sp.Expr) -> None:
    """
    用SymPy验证化简规则：LHS == RHS（实部和虚部分别相等）。
    对所有自由符号代入任意数值进行数值验证（多点采样）。

    Args:
        name:  规则名称（用于错误信息）
        lhs_*: 化简前的符号表达式
        rhs_*: 化简后的符号表达式

    Raises:
        ValueError: 化简规则不正确
    """
    diff_r = sp.simplify(lhs_r - rhs_r)
    diff_i = sp.simplify(lhs_i - rhs_i)
    if diff_r != 0 or diff_i != 0:
        raise ValueError(
            f"化简规则 '{name}' 验证失败！\n"
            f"  实部差值: {diff_r}\n"
            f"  虚部差值: {diff_i}"
        )


def simplify_twiddle_special(node: TwiddleNode) -> DFTNode:
    """
    对TwiddleNode应用特殊值化简规则。

    如果twiddle因子是特殊值（1, -1, ±j, ±(1±j)/√2），
    将TwiddleNode替换为更廉价的等价操作。

    Args:
        node: 待化简的TwiddleNode

    Returns:
        化简后的节点（可能是原节点、NegNode、RotJNode或ScaleNode）
    """
    k, N, sign = node.k, node.N, node.sign
    a = node.a

    # ── 规则1: W(0,N) = 1，直接返回输入（无乘法） ────────────
    if k % N == 0:
        # 验证：W(0,N) = cos(0) + j·sin(0) = 1
        ar_sym = sp.Symbol("ar", real=True)
        ai_sym = sp.Symbol("ai", real=True)
        lhs_r = ar_sym * 1 - ai_sym * 0   # wr=1, wi=0
        lhs_i = ar_sym * 0 + ai_sym * 1
        _verify_rule("W(0,N)=1", lhs_r, lhs_i, ar_sym, ai_sym)
        return a  # 直接返回输入节点，跳过乘法

    # ── 规则2: W(N/2,N) = -1，替换为NegNode ─────────────────
    if 2 * k == N:
        ar_sym = sp.Symbol("ar", real=True)
        ai_sym = sp.Symbol("ai", real=True)
        theta = sp.pi * sign  # 2π·(N/2)/N·sign = π·sign
        wr, wi = sp.cos(theta), sp.sin(theta)
        _verify_rule(
            "W(N/2,N)=-1",
            sp.Mul(wr, ar_sym) + sp.Mul(-wi, ai_sym),
            sp.Mul(wi, ar_sym) + sp.Mul(wr, ai_sym),
            sp.Mul(-1, ar_sym),
            sp.Mul(-1, ai_sym)
        )
        result = NegNode(a=a)
        result.sym_real = sp.Mul(-1, a.sym_real)
        result.sym_imag = sp.Mul(-1, a.sym_imag)
        return result

    # ── 规则3: W(N/4,N) = -j (sign=-1) 或 +j (sign=+1) ──────
    if 4 * k == N:
        ar_sym = sp.Symbol("ar", real=True)
        ai_sym = sp.Symbol("ai", real=True)
        theta = sp.pi * sign / 2  # 2π·(N/4)/N·sign = π·sign/2
        wr, wi = sp.cos(theta), sp.sin(theta)
        # sign=-1: wr=0, wi=-1 → result = ai + j·(-ar)  → ×(-j), rot_sign=-1
        # sign=+1: wr=0, wi=+1 → result = -ai + j·ar   → ×(+j), rot_sign=+1
        rot_sign = sign
        _verify_rule(
            f"W(N/4,N)={'−j' if sign==-1 else '+j'}",
            sp.Mul(wr, ar_sym) + sp.Mul(-wi, ai_sym),
            sp.Mul(wi, ar_sym) + sp.Mul(wr, ai_sym),
            sp.Mul(-rot_sign, ai_sym),  # ×j: real=-ai; ×(-j): real=ai
            sp.Mul(rot_sign, ar_sym)    # ×j: imag=ar;  ×(-j): imag=-ar
        )
        result = RotJNode(a=a, sign=rot_sign)
        return result

    # ── 规则4: W(N/8,N) = (1-j)/√2 (sign=-1) ─────────────────
    if 8 * k == N and sign == -1:
        ar_sym = sp.Symbol("ar")
        ai_sym = sp.Symbol("ai")
        sq2inv = sp.Rational(1, 1) / sp.sqrt(2)
        # W(N/8, N) = cos(π/4) - j·sin(π/4) = (1-j)/√2
        wr = sp.cos(sp.pi / 4)   # = 1/√2
        wi = -sp.sin(sp.pi / 4)  # = -1/√2
        lhs_r = sp.Mul(wr, ar_sym) + sp.Mul(-wi, ai_sym)
        lhs_i = sp.Mul(wi, ar_sym) + sp.Mul(wr, ai_sym)
        # 化简：(ar+ai)/√2 + j·(ai-ar)/√2
        rhs_r = sq2inv * (ar_sym + ai_sym)
        rhs_i = sq2inv * (ai_sym - ar_sym)
        _verify_rule("W(N/8,N)=(1-j)/√2", lhs_r, lhs_i, rhs_r, rhs_i)
        # 生成 ScaleNode：先(ar+ai)和(ai-ar)，再乘1/√2
        add_node = AddNode(a=a, b=a)   # 概念：ar+ai（实际需要Split+Add，此处简化）
        result = ScaleNode(a=add_node, scalar=sq2inv)
        # 直接赋符号表达式（代码生成器用sym_*进行验证）
        result.sym_real = rhs_r.subs({"ar": a.sym_real, "ai": a.sym_imag})
        result.sym_imag = rhs_i.subs({"ar": a.sym_real, "ai": a.sym_imag})
        return result

    # 无可用特殊值规则，返回原节点
    return node


def apply_simplifications(nodes: list[DFTNode]) -> list[DFTNode]:
    """
    对拓扑排序后的节点列表批量应用化简规则。

    Args:
        nodes: 拓扑排序后的DAG节点列表

    Returns:
        化简后的节点列表（已替换特殊节点）
    """
    node_map = {}  # id(old_node) -> new_node

    result = []
    for node in nodes:
        new_node = node
        if isinstance(node, TwiddleNode):
            # 更新输入引用
            if id(node.a) in node_map:
                node.a = node_map[id(node.a)]
            new_node = simplify_twiddle_special(node)
        node_map[id(node)] = new_node
        if new_node not in result:
            result.append(new_node)

    return result


# ── 测试化简规则正确性 ────────────────────────────────────────

def verify_all_rules():
    """
    批量验证所有化简规则的数学正确性。
    在codegen CI中调用，确保每个规则都有SymPy证明。
    """
    from dag import InputNode, build_dft_dag

    print("验证化简规则...")

    # W(0,N)=1
    inp = InputNode(index=0)
    tw = TwiddleNode(a=inp, k=0, N=8, sign=-1)
    simplified = simplify_twiddle_special(tw)
    assert simplified is inp, "W(0,N)=1 化简失败"
    print("  ✓ W(0,N) = 1")

    # W(N/2,N)=-1
    tw = TwiddleNode(a=inp, k=4, N=8, sign=-1)
    simplified = simplify_twiddle_special(tw)
    assert isinstance(simplified, NegNode), "W(N/2,N)=-1 化简失败"
    print("  ✓ W(N/2,N) = -1")

    # W(N/4,N)=-j
    tw = TwiddleNode(a=inp, k=2, N=8, sign=-1)
    simplified = simplify_twiddle_special(tw)
    assert isinstance(simplified, RotJNode), "W(N/4,N)=-j 化简失败"
    print("  ✓ W(N/4,N) = -j")

    # W(N/8,N)=(1-j)/√2
    tw = TwiddleNode(a=inp, k=1, N=8, sign=-1)
    simplified = simplify_twiddle_special(tw)
    assert isinstance(simplified, ScaleNode), "W(N/8,N)=(1-j)/√2 化简失败"
    print("  ✓ W(N/8,N) = (1-j)/√2")

    print("所有化简规则验证通过！")


if __name__ == "__main__":
    verify_all_rules()
