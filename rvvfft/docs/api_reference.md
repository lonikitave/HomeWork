# rvvfft API参考文档

> RISC-V向量(RVV)高性能FFT库 — v0.1.0

## 概述

rvvfft是为贵司RISC-V CPU芯片专用算子库设计的高性能FFT模块，
参考FFTW的codelet+planner+executor分层架构，性能目标对标Intel MKL。

**特性：**
- 支持VLEN=256和VLEN=1024两种向量宽度特化
- 支持交错(interleaved)和分离(split)双数据布局
- 支持f32（单精度）和f64（双精度）
- radix-2/4/8 butterfly kernel（手写RVV + 代码生成器）
- Planner自动选择最优分解策略（ESTIMATE/MEASURE两种模式）
- Wisdom缓存机制（序列化最优plan避免重复测量）

---

## 快速上手

### 交错布局（Interleaved）C2C FFT

```c
#include <rvvfft/rvvfft.h>

int main(void) {
    const int N = 1024;
    
    // 初始化库（自动检测VLEN）
    rvvfft_init(RVVFFT_VLEN_AUTO);
    
    // 分配对齐缓冲区
    rvvfft_complex *in  = rvvfft_aligned_malloc(N * sizeof(rvvfft_complex));
    rvvfft_complex *out = rvvfft_aligned_malloc(N * sizeof(rvvfft_complex));
    
    // 填入数据...
    
    // 创建plan（ESTIMATE模式，快速启发式选择）
    rvvfft_plan plan = rvvfft_plan_dft_1d(N, in, out, RVVFFT_FORWARD, RVVFFT_ESTIMATE);
    
    // 执行FFT
    rvvfft_execute_dft(plan, in, out);
    
    // 清理
    rvvfft_destroy_plan(plan);
    rvvfft_aligned_free(in);
    rvvfft_aligned_free(out);
    rvvfft_cleanup();
    return 0;
}
```

### 分离布局（Split）C2C FFT（最优RVV路径）

```c
// 分离布局：实部虚部分开存储，RVV向量load全部连续访问
float *real_in = rvvfft_aligned_malloc(N * sizeof(float));
float *imag_in = rvvfft_aligned_malloc(N * sizeof(float));
float *real_out = rvvfft_aligned_malloc(N * sizeof(float));
float *imag_out = rvvfft_aligned_malloc(N * sizeof(float));

rvvfft_plan plan = rvvfft_plan_dft_split_1d(
    N, real_in, imag_in, real_out, imag_out,
    RVVFFT_FORWARD, RVVFFT_ESTIMATE);

rvvfft_execute_dft_split(plan, real_in, imag_in, real_out, imag_out);
```

---

## API参考

### Plan创建

#### `rvvfft_plan_dft_1d`
```c
rvvfft_plan rvvfft_plan_dft_1d(
    int             n,      // FFT长度（2的幂，1 ≤ log₂(n) ≤ 24）
    rvvfft_complex *in,     // 输入缓冲区（交错复数）
    rvvfft_complex *out,    // 输出缓冲区（可与in相同实现in-place）
    int             sign,   // RVVFFT_FORWARD(-1) 或 RVVFFT_BACKWARD(+1)
    unsigned        flags   // RVVFFT_ESTIMATE | RVVFFT_MEASURE | ...
);
```

**说明：** 为1D复数FFT（交错布局）创建执行plan。
- `RVVFFT_ESTIMATE`：启发式快速选plan，适合一次性调用
- `RVVFFT_MEASURE`：实际计时所有候选plan，选最快的（开销高，结果可缓存到wisdom）
- 失败返回`NULL`（n不是2的幂、内存不足等）

---

#### `rvvfft_plan_dft_split_1d`
```c
rvvfft_plan rvvfft_plan_dft_split_1d(
    int         n,
    const float *ri, const float *ii,   // 输入实部、虚部
    float       *ro, float       *io,   // 输出实部、虚部
    int          sign,
    unsigned     flags
);
```

**说明：** 分离布局版本，RVV向量化效率最高（推荐使用）。

---

#### `rvvfft_plan_dft_r2c_1d`
```c
rvvfft_plan rvvfft_plan_dft_r2c_1d(
    int             n,      // 实数输入点数（2的幂）
    const float    *in,     // 实数输入（长度≥n）
    rvvfft_complex *out,    // 复数输出（长度≥n/2+1）
    unsigned        flags
);
```

**说明：** 实数FFT（R2C），利用共轭对称性，输出只有n/2+1个频点。

---

### 执行

#### `rvvfft_execute`
```c
void rvvfft_execute(const rvvfft_plan plan);
```
使用plan创建时的缓冲区执行FFT。

#### `rvvfft_execute_dft`
```c
void rvvfft_execute_dft(const rvvfft_plan plan,
                         rvvfft_complex *in, rvvfft_complex *out);
```
用新缓冲区执行（避免重新创建plan的开销，新缓冲区大小和对齐须相同）。

#### `rvvfft_execute_dft_split`
```c
void rvvfft_execute_dft_split(const rvvfft_plan plan,
                               const float *ri, const float *ii,
                               float *ro, float *io);
```
分离布局版本的新缓冲区执行。

---

### Plan管理

#### `rvvfft_destroy_plan`
```c
void rvvfft_destroy_plan(rvvfft_plan plan);
```
释放plan及其所有关联资源（plan=NULL时无操作）。

---

### Wisdom缓存

```c
// 导出wisdom到文件（MEASURE模式测量结果持久化）
rvvfft_status_t rvvfft_export_wisdom_to_file(const char *filename);

// 从文件加载wisdom
rvvfft_status_t rvvfft_import_wisdom_from_file(const char *filename);

// 清除内存中所有wisdom
void rvvfft_forget_wisdom(void);
```

**Wisdom文件格式（文本）：**
```
# rvvfft wisdom v1 VLEN=256
n=1024 sign=-1 layout=1 dtype=0 vlen=256 radix_seq=8,8,4 time_us=12.300
n=4096 sign=-1 layout=1 dtype=0 vlen=256 radix_seq=8,8,8,8 time_us=55.700
```

---

### 初始化与清理

```c
// 初始化：检测VLEN，预分配twiddle表
void rvvfft_init(rvvfft_vlen_t vlen);   // RVVFFT_VLEN_AUTO=0自动检测

// 清理：释放所有全局资源
void rvvfft_cleanup(void);

// 版本信息
const char *rvvfft_version(void);  // 返回如 "0.1.0-VLEN256"
int  rvvfft_get_vlen(void);        // 返回当前VLEN（位）
```

---

### 内存工具

```c
// 分配RVVFFT_ALIGN_BYTES（128字节）对齐的内存
void *rvvfft_aligned_malloc(size_t size);
void *rvvfft_aligned_calloc(size_t count, size_t size);
void  rvvfft_aligned_free(void *ptr);

// 检查指针是否对齐
int rvvfft_is_aligned(const void *ptr);
```

---

## 常量与宏

| 宏/常量 | 值 | 说明 |
|--------|-----|------|
| `RVVFFT_FORWARD` | -1 | 正向DFT，exp(-2πi) |
| `RVVFFT_BACKWARD` | +1 | 逆向DFT，exp(+2πi) |
| `RVVFFT_ESTIMATE` | 0 | 快速启发式plan |
| `RVVFFT_MEASURE` | 1 | 实测选最优plan |
| `RVVFFT_ALIGN_BYTES` | 128 | 内存对齐字节数 |
| `RVVFFT_VLEN` | 256 | 编译时向量宽度（可覆盖） |
| `RVVFFT_IS_POW2(n)` | — | 检查n是否为2的幂 |

---

## 精度保证

| 精度 | 最大相对误差 | 最大ULP误差 | 参考实现 |
|------|------------|-----------|---------|
| f32 | ≤ 1e-6 | ≤ 4 ULP | NumPy FFT（f64计算后截断） |
| f64 | ≤ 1e-14 | ≤ 4 ULP | NumPy FFT（f64） |

---

## 数据布局说明

### 交错布局（Interleaved）
```
内存: [r0, i0, r1, i1, r2, i2, ..., r_{n-1}, i_{n-1}]
类型: rvvfft_complex[]  (struct { float r; float i; })
API:  rvvfft_plan_dft_1d(), rvvfft_execute_dft()
适用: 与外部库（如FFTW、cuFFT）数据格式兼容
```

### 分离布局（Split）
```
内存: real[n] + imag[n] 两个独立数组
      real: [r0, r1, r2, ..., r_{n-1}]
      imag: [i0, i1, i2, ..., i_{n-1}]
API:  rvvfft_plan_dft_split_1d(), rvvfft_execute_dft_split()
适用: RVV向量load全为连续访问，性能最优（推荐）
```
