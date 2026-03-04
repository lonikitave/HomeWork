# Codelet代码生成器使用指南

> rvvfft codegen — Python代码生成器，用于批量产出RVV butterfly kernel

---

## 概述

代码生成器（Phase 2）自动产出所有kernel变体，替代手写kernel：

```
参数组合矩阵：
  radix × layout × dtype × vlen
= [2,4,8] × [split,interleaved] × [f32,f64] × [256,1024]
= 3 × 2 × 2 × 2 = 24个变体（Phase 1目标）

Phase 2完成后扩展到 [2,4,8,16,32] × ... = 60+个变体
```

---

## 文件结构

```
codegen/
├── dag.py          # DFT计算图（DAG）构建：节点类型、拓扑排序
├── simplify.py     # 符号化简pass + SymPy数学验证
├── generator.py    # 主生成脚本：遍历DAG，渲染Jinja2模板
├── verify_math.py  # 独立数学验证脚本（CI门控）
└── templates/
    ├── butterfly.c.j2  # butterfly codelet主模板
    └── twiddle.c.j2    # twiddle因子加载代码片段模板
```

---

## 使用方法

### 环境准备

```bash
pip install sympy jinja2 numpy
```

### 生成单个变体

```bash
cd rvvfft/codegen
python generator.py --radix 4 --layout split --dtype f32 --vlen 256
# 输出：src/kernel/generated/r4_split_f32_vl256.c
```

### 生成所有变体

```bash
python generator.py --all
# 输出：src/kernel/generated/*.c + codelet_registry.h
```

### 生成到自定义目录

```bash
python generator.py --all --output-dir /tmp/my_kernels
```

### 验证数学正确性

```bash
python verify_math.py
# 输出：
# ✓ Cooley-Tukey FFT与DFT定义一致（n=2到1024）
# ✓ IDFT(DFT(x)) = x
# ✓ Radix-4 butterfly代数正确性（SymPy符号验证）
# ✓ W(0,N) = 1, W(N/2,N) = -1, W(N/4,N) = -j, W(N/8,N) = (1-j)/√2
# ✓ 实数输入FFT满足共轭对称性
```

### 验证化简规则

```bash
python simplify.py
# 验证所有twiddle特殊值化简规则
```

---

## 化简规则说明

| 规则 | 数学等式 | 代码效果 | 节省运算 |
|------|---------|---------|---------|
| W(0,N)=1 | exp(0)=1 | 恒等映射，无乘法 | 4 FLOP |
| W(N/2,N)=-1 | exp(jπ)=-1 | NegNode（符号翻转） | 2 FLOP |
| W(N/4,N)=-j | exp(jπ/2)=-j | RotJNode（swap+neg） | 2 FLOP |
| W(N/8,N)=(1-j)/√2 | exp(jπ/4)=... | ScaleNode（1次实数乘） | 3 FLOP |

每条规则在`simplify.py`中均有SymPy符号验证，`verify_math.py`中有数值验证。

---

## 添加新的化简规则

1. 在`simplify.py`的`simplify_twiddle_special()`中添加新的`if`分支
2. 用SymPy验证规则正确性（必须通过`_verify_rule()`）
3. 在`verify_math.py`的`test_twiddle_special_values()`中添加对应测试
4. 运行`python verify_math.py`确认通过

---

## 添加新的Radix支持

1. 在`dag.py`的`build_dft_dag()`中扩展基本情况
2. 更新`generator.py`中的`ALL_VARIANTS`列表
3. 在`src/kernel/handwritten/`中提供对应的标量参考实现
4. 运行生成器产出新kernel
5. 在CMakeLists.txt中确认`file(GLOB)`可以自动收集新文件

---

## AI指令扩展口

生成器设计为可插拔指令后端：

```python
# generator.py
cfg.custom_ext = "none"        # 当前：标准RVV 1.0
cfg.custom_ext = "rvv_ai_ext"  # 未来：贵司AI融合指令
```

新增AI指令支持步骤：
1. 在`generator.py`的`_generate_node_code()`中添加新的代码生成分支
2. 更新Jinja2模板（`butterfly.c.j2`）添加AI指令的`#ifdef`区块
3. 在toolchain配置中添加AI指令的`-march`扩展标志

---

## 与CMake集成

```cmake
# 构建前自动运行代码生成器（可选）
add_custom_target(generate_kernels
    COMMAND python3 ${CMAKE_SOURCE_DIR}/codegen/generator.py --all
            --output-dir ${CMAKE_SOURCE_DIR}/src/kernel/generated
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/codegen
    COMMENT "生成FFT kernel变体..."
)
add_dependencies(rvvfft generate_kernels)
```
