"""
dag.py — DFT计算图（DAG）构建模块

将DFT的数学定义表示为有向无环图（DAG），
DAG节点代表基本运算（Load, Store, Add, Sub, MulComplex, TwiddleMul），
边代表数据流依赖。

代码生成器（generator.py）通过遍历DAG来生成RVV intrinsic C代码。

支持的节点类型：
  - InputNode:  加载输入复数（实部/虚部）
  - OutputNode: 写回输出
  - AddNode:    复数加法  (a + b)
  - SubNode:    复数减法  (a - b)
  - NegNode:    复数取负  (-a)
  - ScaleNode:  实数标量乘（×1/√2等特殊值，不引入额外乘法指令）
  - TwiddleNode:乘以一般twiddle因子W^k（需要复数乘法，最昂贵）
  - RotJNode:   乘以±j（虚数单位，仅需两次加减和符号翻转）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import sympy as sp


# ── 节点基类 ──────────────────────────────────────────────────
class DFTNode:
    """DAG节点基类。每个节点携带符号化的(real, imag)表达式，
    用于SymPy验证；同时记录C代码生成所需的变量名。"""

    _id_counter = 0

    def __init__(self):
        DFTNode._id_counter += 1
        self.node_id: int = DFTNode._id_counter
        # 符号表达式（SymPy）
        self.sym_real: Optional[sp.Expr] = None
        self.sym_imag: Optional[sp.Expr] = None
        # C代码变量名
        self.var_real: str = f"v{self.node_id}_r"
        self.var_imag: str = f"v{self.node_id}_i"
        # 后继节点列表（用于拓扑排序）
        self.consumers: List[DFTNode] = []


@dataclass
class InputNode(DFTNode):
    """加载第k个输入复数 X[k]。"""
    index: int = 0           # 输入索引k
    layout: str = "split"    # "split" | "interleaved"

    def __post_init__(self):
        super().__init__()
        k = self.index
        # 符号变量：x_k_r + j·x_k_i
        self.sym_real = sp.Symbol(f"x_{k}_r")
        self.sym_imag = sp.Symbol(f"x_{k}_i")


@dataclass
class AddNode(DFTNode):
    """复数加法：out = a + b"""
    a: DFTNode = field(default=None)
    b: DFTNode = field(default=None)

    def __post_init__(self):
        super().__init__()
        if self.a and self.b:
            self.sym_real = sp.Add(self.a.sym_real, self.b.sym_real)
            self.sym_imag = sp.Add(self.a.sym_imag, self.b.sym_imag)
            self.a.consumers.append(self)
            self.b.consumers.append(self)


@dataclass
class SubNode(DFTNode):
    """复数减法：out = a - b"""
    a: DFTNode = field(default=None)
    b: DFTNode = field(default=None)

    def __post_init__(self):
        super().__init__()
        if self.a and self.b:
            self.sym_real = sp.Add(self.a.sym_real, sp.Mul(-1, self.b.sym_real))
            self.sym_imag = sp.Add(self.a.sym_imag, sp.Mul(-1, self.b.sym_imag))
            self.a.consumers.append(self)
            self.b.consumers.append(self)


@dataclass
class NegNode(DFTNode):
    """复数取负：out = -a"""
    a: DFTNode = field(default=None)

    def __post_init__(self):
        super().__init__()
        if self.a:
            self.sym_real = sp.Mul(-1, self.a.sym_real)
            self.sym_imag = sp.Mul(-1, self.a.sym_imag)
            self.a.consumers.append(self)


@dataclass
class RotJNode(DFTNode):
    """乘以±j（虚数单位旋转）：
      +j: (a_r + j·a_i) × j = -a_i + j·a_r   [swap + neg]
      -j: (a_r + j·a_i) × (-j) = a_i - j·a_r  [swap + neg]
    仅需加减和符号翻转，无乘法指令。"""
    a: DFTNode = field(default=None)
    sign: int = 1  # +1 表示 ×j，-1 表示 ×(-j)

    def __post_init__(self):
        super().__init__()
        if self.a:
            if self.sign == 1:   # ×j
                self.sym_real = sp.Mul(-1, self.a.sym_imag)
                self.sym_imag = self.a.sym_real
            else:                # ×(-j)
                self.sym_real = self.a.sym_imag
                self.sym_imag = sp.Mul(-1, self.a.sym_real)
            self.a.consumers.append(self)


@dataclass
class ScaleNode(DFTNode):
    """实数标量乘：out = scalar × a（如 ×1/√2，只用一次实数乘） """
    a: DFTNode = field(default=None)
    scalar: sp.Expr = field(default_factory=lambda: sp.Integer(1))

    def __post_init__(self):
        super().__init__()
        if self.a:
            self.sym_real = sp.Mul(self.scalar, self.a.sym_real)
            self.sym_imag = sp.Mul(self.scalar, self.a.sym_imag)
            self.a.consumers.append(self)


@dataclass
class TwiddleNode(DFTNode):
    """乘以一般twiddle因子 W(k,N) = exp(-2πj·k/N)：
       out = W·a = (wr+j·wi)·(ar+j·ai) = (wr·ar - wi·ai) + j·(wi·ar + wr·ai)
    需要两次实数乘法和一次复数乘法（共4次乘法，可用FMA优化为3次）。
    SymPy验证时替换为具体数值。"""
    a: DFTNode = field(default=None)
    k: int = 0     # twiddle指数
    N: int = 1     # DFT长度（用于计算W(k,N)）
    sign: int = -1 # -1: 正向DFT, +1: 逆向

    def __post_init__(self):
        super().__init__()
        if self.a:
            # 符号化twiddle：cos(2π·k/N) - j·sign·sin(2π·k/N)
            theta = sp.Rational(2 * self.k, self.N) * sp.pi * self.sign
            wr = sp.cos(theta)
            wi = sp.sin(theta)
            self.sym_real = sp.Add(
                sp.Mul(wr, self.a.sym_real),
                sp.Mul(-wi, self.a.sym_imag))
            self.sym_imag = sp.Add(
                sp.Mul(wi, self.a.sym_real),
                sp.Mul(wr, self.a.sym_imag))
            self.a.consumers.append(self)


@dataclass
class OutputNode(DFTNode):
    """FFT输出节点，标记某个节点的结果为第index个输出。"""
    src: DFTNode = field(default=None)
    index: int = 0

    def __post_init__(self):
        super().__init__()
        if self.src:
            self.sym_real = self.src.sym_real
            self.sym_imag = self.src.sym_imag
            self.src.consumers.append(self)


# ── DAG构建辅助 ───────────────────────────────────────────────

def build_dft_dag(n: int, sign: int = -1) -> Tuple[List[InputNode], List[OutputNode]]:
    """
    构建长度n的DFT的完整计算DAG（Cooley-Tukey DIT递归分解）。

    仅用于小n（codelet级别），大n由生成器递归调用。

    Args:
        n:    DFT长度（2的幂，建议≤8用于codelet生成）
        sign: -1（正向），+1（逆向）

    Returns:
        (inputs, outputs): 输入节点列表和输出节点列表
    """
    inputs = [InputNode(index=k) for k in range(n)]

    if n == 1:
        outputs = [OutputNode(src=inputs[0], index=0)]
        return inputs, outputs

    # 递归分解：DIT radix-2
    # 偶数索引输入做n/2点DFT，奇数索引输入做n/2点DFT，再合并
    half = n // 2
    even_in = [inputs[2*k] for k in range(half)]
    odd_in  = [inputs[2*k+1] for k in range(half)]

    # 递归构建子DAG（直接使用InputNode，不嵌套子函数，保持DAG可达性）
    # 简化：对叶子节点（n=2）直接手工展开
    if n == 2:
        tw = TwiddleNode(a=odd_in[0], k=0, N=2, sign=sign)  # W(0,2)=1，会被simplify
        out0 = AddNode(a=even_in[0], b=tw)
        out1 = SubNode(a=even_in[0], b=tw)
        outputs = [OutputNode(src=out0, index=0),
                   OutputNode(src=out1, index=1)]
        return inputs, outputs

    # n>2：递归组合（此处返回合并后的输出节点）
    _, even_out = build_dft_dag(half, sign)
    _, odd_out  = build_dft_dag(half, sign)

    outputs = []
    for k in range(half):
        tw = TwiddleNode(a=odd_out[k].src, k=k, N=n, sign=sign)
        xk      = AddNode(a=even_out[k].src, b=tw)
        xk_half = SubNode(a=even_out[k].src, b=tw)
        outputs.append(OutputNode(src=xk,      index=k))
        outputs.append(OutputNode(src=xk_half, index=k + half))

    return inputs, outputs


def topological_sort(outputs: List[OutputNode]) -> List[DFTNode]:
    """对DAG做拓扑排序，返回按计算顺序排列的节点列表。"""
    visited = set()
    order = []

    def dfs(node: DFTNode):
        if id(node) in visited:
            return
        visited.add(id(node))
        # 访问所有输入节点（前驱）
        for attr in ['a', 'b', 'src']:
            pred = getattr(node, attr, None)
            if pred is not None:
                dfs(pred)
        order.append(node)

    for out in outputs:
        dfs(out)

    return order
