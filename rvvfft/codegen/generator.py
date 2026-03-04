"""
generator.py — Codelet代码生成器（主脚本）

从参数配置生成RVV intrinsic C代码的完整流程：
  1. 构建DFT计算DAG（dag.py）
  2. 符号化简（simplify.py）
  3. SymPy数学验证（verify_math.py中的单元函数）
  4. Jinja2模板渲染输出C代码
  5. 批量生成并写入 src/kernel/generated/ 目录

使用方法：
  python generator.py --radix 4 --layout split --dtype f32 --vlen 256
  python generator.py --all    # 生成所有组合变体

生成器输出示例（radix-2, split, f32）：
  src/kernel/generated/r2_split_f32_vl256.c
  src/kernel/generated/r4_split_f32_vl256.c
  src/kernel/generated/r8_split_f32_vl256.c
  ...（共~60+文件，Phase 2完成后）
"""
from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

# 确保codegen目录在PYTHONPATH中
sys.path.insert(0, os.path.dirname(__file__))

try:
    from jinja2 import Environment, FileSystemLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    print("[警告] Jinja2未安装，使用内置字符串模板作为回退")

from dag import DFTNode, InputNode, OutputNode, TwiddleNode, build_dft_dag, topological_sort
from simplify import apply_simplifications


# ── 生成器配置 ─────────────────────────────────────────────────

@dataclass
class CodegenConfig:
    radix:  int   = 2          # 支持: 2, 4, 8, 16, 32
    layout: str   = "split"    # "split" | "interleaved"
    dtype:  str   = "f32"      # "f32" | "f64"
    vlen:   int   = 256        # 256 | 1024
    lmul:   int   = 1          # LMUL寄存器分组: 1, 2, 4, 8
    sign:   int   = -1         # -1: FORWARD, +1: BACKWARD
    # 扩展口（预留AI指令）
    custom_ext: str = "none"   # "none" | "rvv_ai_ext"（未来扩展）
    # 输出目录
    output_dir: str = os.path.join(
        os.path.dirname(__file__), "..", "src", "kernel", "generated")


# ── 所有需要生成的变体组合 ────────────────────────────────────

ALL_VARIANTS = [
    CodegenConfig(radix=r, layout=l, dtype=d, vlen=v)
    for r in [2, 4, 8]
    for l in ["split", "interleaved"]
    for d in ["f32", "f64"]
    for v in [256, 1024]
]


# ── 代码生成核心 ──────────────────────────────────────────────

def _c_type(dtype: str) -> str:
    return "float" if dtype == "f32" else "double"


def _rvv_sew(dtype: str) -> str:
    return "32" if dtype == "f32" else "64"


def _generate_node_code(node: DFTNode, dtype: str, layout: str) -> str:
    """将单个DAG节点转换为RVV intrinsic C语句（骨架，Phase 2完整实现）。"""
    sew = _rvv_sew(dtype)
    ctype = _c_type(dtype)
    vr, vi = node.var_real, node.var_imag

    if isinstance(node, InputNode):
        k = node.index
        if layout == "split":
            return (f"    vfloat{sew}m1_t {vr} = __riscv_vle{sew}_v_f{sew}m1(r_in + {k} * stride, vl);\n"
                    f"    vfloat{sew}m1_t {vi} = __riscv_vle{sew}_v_f{sew}m1(i_in + {k} * stride, vl);")
        else:
            return (f"    // interleaved: vlseg2e{sew} for element {k}\n"
                    f"    vfloat{sew}m1_t {vr} = /* real part of X[{k}] */;\n"
                    f"    vfloat{sew}m1_t {vi} = /* imag part of X[{k}] */;")

    elif isinstance(node, OutputNode) and node.src:
        k = node.index
        sr, si = node.src.var_real, node.src.var_imag
        if layout == "split":
            return (f"    __riscv_vse{sew}_v_f{sew}m1(r_out + {k} * stride, {sr}, vl);\n"
                    f"    __riscv_vse{sew}_v_f{sew}m1(i_out + {k} * stride, {si}, vl);")
        return f"    /* output[{k}]: {sr}, {si} */"

    elif hasattr(node, 'a') and hasattr(node, 'b') and \
         node.__class__.__name__ == 'AddNode':
        ar, ai = node.a.var_real, node.a.var_imag
        br, bi = node.b.var_real, node.b.var_imag
        return (f"    vfloat{sew}m1_t {vr} = __riscv_vfadd_vv_f{sew}m1({ar}, {br}, vl);\n"
                f"    vfloat{sew}m1_t {vi} = __riscv_vfadd_vv_f{sew}m1({ai}, {bi}, vl);")

    elif hasattr(node, 'a') and hasattr(node, 'b') and \
         node.__class__.__name__ == 'SubNode':
        ar, ai = node.a.var_real, node.a.var_imag
        br, bi = node.b.var_real, node.b.var_imag
        return (f"    vfloat{sew}m1_t {vr} = __riscv_vfsub_vv_f{sew}m1({ar}, {br}, vl);\n"
                f"    vfloat{sew}m1_t {vi} = __riscv_vfsub_vv_f{sew}m1({ai}, {bi}, vl);")

    elif node.__class__.__name__ == 'RotJNode':
        ar, ai = node.a.var_real, node.a.var_imag
        rot = getattr(node, 'sign', 1)
        if rot == 1:   # ×j: real=-ai, imag=ar
            return (f"    vfloat{sew}m1_t {vr} = __riscv_vfneg_v_f{sew}m1({ai}, vl);\n"
                    f"    vfloat{sew}m1_t {vi} = {ar};  /* ×j */")
        else:          # ×(-j): real=ai, imag=-ar
            return (f"    vfloat{sew}m1_t {vr} = {ai};  /* ×(-j) */\n"
                    f"    vfloat{sew}m1_t {vi} = __riscv_vfneg_v_f{sew}m1({ar}, vl);")

    elif isinstance(node, TwiddleNode):
        ar, ai = node.a.var_real, node.a.var_imag
        return (
            f"    /* TwiddleNode W({node.k},{node.N}) */\n"
            f"    vfloat{sew}m1_t {vr} = __riscv_vfmul_vv_f{sew}m1(tw_{node.k}r, {ar}, vl);\n"
            f"    {vr} = __riscv_vfnmsac_vv_f{sew}m1({vr}, tw_{node.k}i, {ai}, vl);\n"
            f"    vfloat{sew}m1_t {vi} = __riscv_vfmul_vv_f{sew}m1(tw_{node.k}i, {ar}, vl);\n"
            f"    {vi} = __riscv_vfmacc_vv_f{sew}m1({vi}, tw_{node.k}r, {ai}, vl);"
        )

    return f"    /* {node.__class__.__name__}: {vr}, {vi} */"


def generate_codelet_c(cfg: CodegenConfig) -> str:
    """
    为给定配置生成完整的C codelet文件内容。

    Phase 1：生成框架骨架（函数签名、注释、向量化循环结构）
    Phase 2：基于DAG拓扑序完整展开butterfly运算
    """
    ctype  = _c_type(cfg.dtype)
    sew    = _rvv_sew(cfg.dtype)
    suffix = f"r{cfg.radix}_{cfg.layout}_{cfg.dtype}_vl{cfg.vlen}"
    fname  = f"rvvfft_{suffix}"
    vl_f32 = cfg.vlen // (32 if cfg.dtype == "f32" else 64)

    # 构建并化简DAG（仅用于小radix的codelet级展开）
    DFTNode._id_counter = 0  # 重置ID计数器
    if cfg.radix <= 8:
        _, outputs = build_dft_dag(cfg.radix, cfg.sign)
        nodes = topological_sort(outputs)
        nodes = apply_simplifications(nodes)
        dag_code = "\n".join(_generate_node_code(n, cfg.dtype, cfg.layout)
                              for n in nodes
                              if not isinstance(n, (InputNode, OutputNode)))
        store_code = "\n".join(_generate_node_code(n, cfg.dtype, cfg.layout)
                                for n in nodes if isinstance(n, OutputNode))
        load_code  = "\n".join(_generate_node_code(n, cfg.dtype, cfg.layout)
                                for n in nodes if isinstance(n, InputNode))
    else:
        dag_code = load_code = store_code = "    /* Phase 2: 更高radix由生成器完整展开 */"

    # 确定RVV头文件条件编译
    rvv_guard = "#ifdef RVVFFT_USE_RVV\n#include <riscv_vector.h>"
    rvv_end   = "#endif /* RVVFFT_USE_RVV */"

    code = f"""\
/**
 * {suffix}.c — 自动生成的FFT codelet
 *
 * 生成参数:
 *   radix:  {cfg.radix}
 *   layout: {cfg.layout}
 *   dtype:  {cfg.dtype}
 *   vlen:   {cfg.vlen}
 *   lmul:   {cfg.lmul}
 *   sign:   {'FORWARD(-1)' if cfg.sign == -1 else 'BACKWARD(+1)'}
 *   custom_ext: {cfg.custom_ext}
 *
 * 每次向量化循环处理 {vl_f32} 个butterfly（VLEN={cfg.vlen}, SEW={sew}, LMUL={cfg.lmul}）。
 * 生成时间: 由generator.py自动产出，请勿手动修改。
 * 修改需求: 更新generator.py模板后重新生成。
 */
#include <stddef.h>
#include "../../../include/rvvfft_types.h"

{rvv_guard}

/**
 * @brief Radix-{cfg.radix} DIT butterfly，{cfg.layout}布局，{cfg.dtype}，VLEN={cfg.vlen}。
 *
 * @param ri    输入实部（{cfg.layout}布局）
 * @param ii    输入虚部
 * @param ro    输出实部（in-place: ro==ri）
 * @param io    输出虚部
 * @param tw    Twiddle factor表（预计算，VLEN={cfg.vlen}对齐）
 * @param stride 步长（元素单位，stride=1时使用向量化路径）
 * @param n     本级DFT长度
 * @param sign  RVVFFT_FORWARD(-1) 或 RVVFFT_BACKWARD(+1)
 */
void {fname}(
    const {ctype} *ri, const {ctype} *ii,
    {ctype}       *ro, {ctype}       *io,
    const {ctype} *tw,
    ptrdiff_t      stride,
    int            n,
    int            sign)
{{
    (void)sign; /* twiddle已按sign预计算 */
    int group = n / {cfg.radix};

    if (stride != 1) {{
        /* 非连续访问：退回到标量实现 */
        extern void rvvfft_r{cfg.radix}_{cfg.layout}_{cfg.dtype}_scalar(
            {ctype} *, {ctype} *, const {ctype} *, int, ptrdiff_t, int);
        rvvfft_r{cfg.radix}_{cfg.layout}_{cfg.dtype}_scalar(
            ({ctype} *)ri, ({ctype} *)ii, tw, n, stride, sign);
        return;
    }}

    int remaining = group, off = 0;
    while (remaining > 0) {{
        size_t vl = __riscv_vsetvl_e{sew}m{cfg.lmul}((size_t)remaining);

        /* ── 加载输入 ── */
{load_code}

        /* ── Butterfly运算（DAG展开） ── */
{dag_code}

        /* ── 写回输出 ── */
{store_code}

        off       += (int)vl;
        remaining -= (int)vl;
    }}
}}

{rvv_end}
"""
    return code


def generate_all(output_dir: str | None = None) -> None:
    """批量生成所有codelet变体并写入文件。"""
    for cfg in ALL_VARIANTS:
        if output_dir:
            cfg.output_dir = output_dir
        os.makedirs(cfg.output_dir, exist_ok=True)

        filename = f"r{cfg.radix}_{cfg.layout}_{cfg.dtype}_vl{cfg.vlen}.c"
        filepath = os.path.join(cfg.output_dir, filename)

        code = generate_codelet_c(cfg)
        with open(filepath, "w") as f:
            f.write(code)
        print(f"生成: {filepath}")

    # 生成注册表头文件
    _generate_registry(output_dir or ALL_VARIANTS[0].output_dir)


def _generate_registry(output_dir: str) -> None:
    """生成 codelet_registry.h，供planner自动发现所有kernel。"""
    lines = [
        "/* codelet_registry.h — 自动生成，勿手动修改 */",
        "#pragma once",
        "#include \"../../../include/rvvfft_types.h\"",
        "",
        "/* 前向声明所有生成的codelet函数 */",
    ]
    for cfg in ALL_VARIANTS:
        suffix = f"r{cfg.radix}_{cfg.layout}_{cfg.dtype}_vl{cfg.vlen}"
        ctype = _c_type(cfg.dtype)
        lines.append(
            f"extern void rvvfft_{suffix}("
            f"const {ctype}*, const {ctype}*, {ctype}*, {ctype}*,"
            f" const {ctype}*, ptrdiff_t, int, int);"
        )
    lines += [
        "",
        "/* Codelet注册表条目（由planner.c中的g_codelet_registry加载） */",
        "#define RVVFFT_REGISTER_GENERATED_CODELETS() do { \\"
    ]
    for cfg in ALL_VARIANTS:
        suffix = f"r{cfg.radix}_{cfg.layout}_{cfg.dtype}_vl{cfg.vlen}"
        lines.append(
            f"    RVVFFT_REGISTER({cfg.radix}, RVVFFT_LAYOUT_{'SPLIT' if cfg.layout == 'split' else 'INTERLEAVED'},"
            f" RVVFFT_DTYPE_{'F32' if cfg.dtype == 'f32' else 'F64'}, {cfg.vlen}, rvvfft_{suffix}); \\"
        )
    lines.append("} while(0)")

    registry_path = os.path.join(output_dir, "codelet_registry.h")
    with open(registry_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"生成注册表: {registry_path}")


# ── 命令行接口 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="rvvfft Codelet代码生成器")
    parser.add_argument("--radix",  type=int, default=2, choices=[2, 4, 8, 16, 32])
    parser.add_argument("--layout", default="split", choices=["split", "interleaved"])
    parser.add_argument("--dtype",  default="f32",   choices=["f32", "f64"])
    parser.add_argument("--vlen",   type=int, default=256, choices=[256, 1024])
    parser.add_argument("--lmul",   type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--sign",   type=int, default=-1, choices=[-1, 1])
    parser.add_argument("--all",    action="store_true", help="生成所有变体")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.all:
        generate_all(args.output_dir)
    else:
        cfg = CodegenConfig(
            radix=args.radix, layout=args.layout,
            dtype=args.dtype, vlen=args.vlen,
            lmul=args.lmul, sign=args.sign)
        if args.output_dir:
            cfg.output_dir = args.output_dir
        os.makedirs(cfg.output_dir, exist_ok=True)
        code = generate_codelet_c(cfg)
        filename = f"r{cfg.radix}_{cfg.layout}_{cfg.dtype}_vl{cfg.vlen}.c"
        filepath = os.path.join(cfg.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(code)
        print(f"生成: {filepath}")
        print(code[:500] + "\n...")


if __name__ == "__main__":
    main()
