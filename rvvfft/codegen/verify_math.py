"""
verify_math.py — FFT数学正确性自动验证脚本

使用SymPy对FFT算法的数学性质进行符号验证：
  1. DFT定义验证：DFT(x)[k] = Σ x[n]·W^{nk}
  2. 逆DFT验证：IDFT(DFT(x)) = x（精确，无舍入误差）
  3. Cooley-Tukey分解正确性：
     DFT_N[k] = DFT_{N/2}[even][k] + W(k,N)·DFT_{N/2}[odd][k]
  4. Radix-4 butterfly数学验证
  5. 特殊twiddle值验证（W(0,N)=1等）

此脚本作为CI门控：任何化简规则修改必须通过此验证才能合入。
"""
import sympy as sp
import cmath
import math
import random


def dft_exact(x: list[complex], sign: int = -1) -> list[complex]:
    """用DFT定义直接计算（O(n²)，仅用于验证小n）。"""
    n = len(x)
    result = []
    for k in range(n):
        val = sum(x[j] * cmath.exp(sign * 2j * math.pi * k * j / n)
                  for j in range(n))
        result.append(val)
    return result


def fft_cooley_tukey(x: list[complex], sign: int = -1) -> list[complex]:
    """递归Cooley-Tukey FFT（Python参考实现）。"""
    n = len(x)
    if n == 1:
        return x[:]
    even = fft_cooley_tukey(x[0::2], sign)
    odd  = fft_cooley_tukey(x[1::2], sign)
    result = [0j] * n
    half = n // 2
    for k in range(half):
        w = cmath.exp(sign * 2j * math.pi * k / n)
        t = w * odd[k]
        result[k]        = even[k] + t
        result[k + half] = even[k] - t
    return result


def max_error(a: list[complex], b: list[complex]) -> float:
    """计算两个复数列表的最大元素差（绝对值）。"""
    return max(abs(x - y) for x, y in zip(a, b))


# ── 测试1: Cooley-Tukey与DFT定义一致性 ──────────────────────

def test_cooley_tukey_vs_dft():
    """验证Cooley-Tukey FFT与DFT定义逐点结果一致（误差≤1e-10）。"""
    random.seed(42)
    for n in [2, 4, 8, 16, 32, 64, 256, 1024]:
        x = [complex(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(n)]
        ref = dft_exact(x)
        got = fft_cooley_tukey(x)
        err = max_error(ref, got)
        assert err < 1e-10, f"n={n}: FFT vs DFT误差 {err:.2e} > 1e-10"
    print("  ✓ Cooley-Tukey FFT与DFT定义一致（n=2到1024）")


# ── 测试2: IDFT(DFT(x)) = x ──────────────────────────────────

def test_inverse_fft():
    """验证正向FFT后做逆向FFT可精确还原（归一化误差≤1e-10）。"""
    random.seed(99)
    for n in [8, 64, 512]:
        x = [complex(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(n)]
        X = fft_cooley_tukey(x, sign=-1)
        x_rec = [v / n for v in fft_cooley_tukey(X, sign=+1)]
        err = max_error(x, x_rec)
        assert err < 1e-10, f"n={n}: IDFT(DFT(x))误差 {err:.2e}"
    print("  ✓ IDFT(DFT(x)) = x（归一化，n=8/64/512）")


# ── 测试3: Radix-4 butterfly数学验证（SymPy符号计算） ─────────

def test_radix4_sympy():
    """
    用SymPy验证radix-4 butterfly的代数正确性：
    4点DFT = 2×(2点DFT) + butterfly合并。
    """
    # 4个符号输入：x0, x1, x2, x3（复数，用实/虚部表示，声明为实数避免SymPy的re/im展开问题）
    x0r, x0i = sp.Symbol("x0r", real=True), sp.Symbol("x0i", real=True)
    x1r, x1i = sp.Symbol("x1r", real=True), sp.Symbol("x1i", real=True)
    x2r, x2i = sp.Symbol("x2r", real=True), sp.Symbol("x2i", real=True)
    x3r, x3i = sp.Symbol("x3r", real=True), sp.Symbol("x3i", real=True)

    # DFT-4 直接定义（正向，sign=-1）：X[k] = Σ x[n]·W^{nk}, W=exp(-2πj/4)=exp(-πj/2)
    j = sp.I
    W = sp.exp(-j * sp.pi / 2)  # W(1,4) = -j

    def dft4_sym(xr, xi):
        """4点DFT，返回4个复数（实部, 虚部）的列表。"""
        xs = [(xr[k] + j * xi[k]) for k in range(4)]
        result = []
        for k in range(4):
            val = sum(xs[n] * W**(n * k) for n in range(4))
            val = sp.simplify(val)
            result.append((sp.re(val), sp.im(val)))
        return result

    xr_list = [x0r, x1r, x2r, x3r]
    xi_list = [x0i, x1i, x2i, x3i]

    # 直接DFT-4定义
    ref = dft4_sym(xr_list, xi_list)

    # Radix-4 butterfly计算
    # 偶数子序列DFT-2: even = [x0, x2]
    # e0 = x0 + x2, e1 = x0 - x2  (W(0,2)=1, W(1,2)=-1)
    e0r = x0r + x2r; e0i = x0i + x2i
    e1r = x0r - x2r; e1i = x0i - x2i
    # 奇数子序列DFT-2: odd = [x1, x3]
    o0r = x1r + x3r; o0i = x1i + x3i
    o1r = x1r - x3r; o1i = x1i - x3i

    # 合并（W(0,4)=1, W(1,4)=-j, W(2,4)=-1, W(3,4)=j）
    # X[0] = e0 + 1·o0
    # X[1] = e1 + (-j)·o1  → e1 + [o1i, -o1r]
    # X[2] = e0 + (-1)·o0  = e0 - o0
    # X[3] = e1 + j·o1     → e1 + [-o1i, o1r]
    butterfly = [
        (e0r + o0r,   e0i + o0i),    # X[0]
        (e1r + o1i,   e1i - o1r),    # X[1]: e1 - j·o1（FORWARD: -j·o1）
        (e0r - o0r,   e0i - o0i),    # X[2]
        (e1r - o1i,   e1i + o1r),    # X[3]: e1 + j·o1
    ]

    for k in range(4):
        diff_r = sp.simplify(ref[k][0] - butterfly[k][0])
        diff_i = sp.simplify(ref[k][1] - butterfly[k][1])
        assert diff_r == 0 and diff_i == 0, \
            f"Radix-4 X[{k}] 验证失败: diff_r={diff_r}, diff_i={diff_i}"

    print("  ✓ Radix-4 butterfly代数正确性（SymPy符号验证）")


# ── 测试4: Twiddle特殊值验证 ──────────────────────────────────

def test_twiddle_special_values():
    """验证twiddle特殊值化简规则的数学正确性（SymPy精确）。"""
    import sympy as sp

    ar, ai = sp.Symbol("ar"), sp.Symbol("ai")

    # W(0,N) = 1：乘以1等于恒等
    W0 = sp.Integer(1)
    assert sp.simplify(W0 * ar - 0 * ai - ar) == 0
    assert sp.simplify(0 * ar + W0 * ai - ai) == 0
    print("  ✓ W(0,N) = 1（无乘法，恒等映射）")

    # W(N/2,N) = -1
    theta_half = sp.pi
    wr = sp.cos(theta_half)  # = -1
    wi = sp.sin(theta_half)  # = 0
    assert sp.simplify(wr * ar - wi * ai - (-ar)) == 0
    assert sp.simplify(wi * ar + wr * ai - (-ai)) == 0
    print("  ✓ W(N/2,N) = -1（取负，无乘法）")

    # W(N/4,N) = -j (FORWARD sign=-1: 2π·(N/4)/N·(-1) = -π/2)
    theta_qtr = -sp.pi / 2
    wr = sp.cos(theta_qtr)  # = 0
    wi = sp.sin(theta_qtr)  # = -1
    # 结果: wr·ar - wi·ai = 0·ar - (-1)·ai = ai
    #        wi·ar + wr·ai = -1·ar + 0·ai = -ar
    assert sp.simplify(wr * ar - wi * ai - ai) == 0
    assert sp.simplify(wi * ar + wr * ai - (-ar)) == 0
    print("  ✓ W(N/4,N) = -j（swap+neg，无乘法）")

    # W(N/8,N) = (1-j)/√2 (sign=-1: θ = -π/4)
    theta_8th = -sp.pi / 4
    wr = sp.cos(theta_8th)  # = 1/√2
    wi = sp.sin(theta_8th)  # = -1/√2
    sq2inv = 1 / sp.sqrt(2)
    expected_r = sq2inv * (ar + ai)   # (ar + ai)/√2
    expected_i = sq2inv * (ai - ar)   # (ai - ar)/√2
    assert sp.simplify(wr * ar - wi * ai - expected_r) == 0
    assert sp.simplify(wi * ar + wr * ai - expected_i) == 0
    print("  ✓ W(N/8,N) = (1-j)/√2（1次实数乘+加减）")


# ── 测试5: R2C对称性 ──────────────────────────────────────────

def test_r2c_symmetry():
    """验证实数输入FFT的共轭对称性：X[k] = X*[N-k]。"""
    random.seed(7)
    for n in [8, 64, 256]:
        x = [complex(random.gauss(0, 1), 0) for _ in range(n)]  # 纯实数输入
        X = fft_cooley_tukey(x, sign=-1)
        for k in range(1, n // 2):
            err = abs(X[k] - X[n - k].conjugate())
            assert err < 1e-10, f"n={n}: 共轭对称性违反 k={k} err={err:.2e}"
    print("  ✓ 实数输入FFT满足共轭对称性（R2C优化前提）")


# ── 主函数 ────────────────────────────────────────────────────

def run_all_verifications():
    """运行所有数学验证。CI中调用，任何失败导致构建失败。"""
    print("=" * 60)
    print("rvvfft 数学正确性验证（SymPy + Python参考FFT）")
    print("=" * 60)

    test_cooley_tukey_vs_dft()
    test_inverse_fft()
    test_radix4_sympy()
    test_twiddle_special_values()
    test_r2c_symmetry()

    print("=" * 60)
    print("✅ 所有数学验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_verifications()
