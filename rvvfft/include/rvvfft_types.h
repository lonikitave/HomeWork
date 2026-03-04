/**
 * rvvfft_types.h — 类型定义、常量与宏
 *
 * 为RISC-V向量(RVV)高性能FFT库提供基础类型和编译时配置宏。
 * 支持VLEN=256和VLEN=1024两种向量长度特化，以及interleaved/split双数据布局。
 */
#ifndef RVVFFT_TYPES_H
#define RVVFFT_TYPES_H

#include <stddef.h>
#include <stdint.h>

/* ── 版本 ─────────────────────────────────────────────────── */
#define RVVFFT_VERSION_MAJOR 0
#define RVVFFT_VERSION_MINOR 1
#define RVVFFT_VERSION_PATCH 0

/* ── FFT方向 ──────────────────────────────────────────────── */
#define RVVFFT_FORWARD  (-1)   /**< 正向DFT：exp(-2πi·k/N) */
#define RVVFFT_BACKWARD (+1)   /**< 逆向DFT：exp(+2πi·k/N) */

/* ── Plan flags ───────────────────────────────────────────── */
#define RVVFFT_ESTIMATE  (0u)          /**< 启发式快速选plan，不做实测 */
#define RVVFFT_MEASURE   (1u << 0)     /**< 实际计时，选最优plan */
#define RVVFFT_PATIENT   (1u << 1)     /**< 更穷举的搜索（慢） */
#define RVVFFT_WISDOM_ONLY (1u << 2)   /**< 仅从wisdom缓存读取plan */

/* ── 数据布局 ─────────────────────────────────────────────── */
typedef enum {
    RVVFFT_LAYOUT_INTERLEAVED = 0, /**< [r0,i0,r1,i1,...] 交错 */
    RVVFFT_LAYOUT_SPLIT       = 1  /**< real[]: [r0,r1,...] + imag[]: [i0,i1,...] 分离 */
} rvvfft_layout_t;

/* ── VLEN特化标识 ─────────────────────────────────────────── */
typedef enum {
    RVVFFT_VLEN_AUTO = 0,    /**< 运行时检测 */
    RVVFFT_VLEN_256  = 256,  /**< 明确指定256位向量宽度 */
    RVVFFT_VLEN_1024 = 1024  /**< 明确指定1024位向量宽度 */
} rvvfft_vlen_t;

/* ── 浮点精度 ─────────────────────────────────────────────── */
typedef enum {
    RVVFFT_DTYPE_F32 = 0, /**< 单精度 float32 */
    RVVFFT_DTYPE_F64 = 1  /**< 双精度 float64 */
} rvvfft_dtype_t;

/* ── 复数类型 ─────────────────────────────────────────────── */
/** 交错布局下的单精度复数 */
typedef struct { float    r; float    i; } rvvfft_complex_f32;
/** 交错布局下的双精度复数 */
typedef struct { double   r; double   i; } rvvfft_complex_f64;

/** 默认使用单精度（与API rvvfft_complex对齐） */
typedef rvvfft_complex_f32 rvvfft_complex;

/* ── Radix类型 ────────────────────────────────────────────── */
typedef enum {
    RVVFFT_RADIX_2  = 2,
    RVVFFT_RADIX_4  = 4,
    RVVFFT_RADIX_8  = 8,
    RVVFFT_RADIX_16 = 16,
    RVVFFT_RADIX_32 = 32
} rvvfft_radix_t;

/* ── 内存对齐要求 ─────────────────────────────────────────── */
/* 按VLEN/8字节对齐，VLEN=1024时需128字节对齐 */
#define RVVFFT_ALIGN_BYTES 128

/* ── 编译时VLEN宏（由工具链或CMake注入） ─────────────────── */
#ifndef RVVFFT_VLEN
#  define RVVFFT_VLEN 256   /* 默认256，可在编译时覆盖 */
#endif

/* 每个向量寄存器可容纳的f32元素数（LMUL=1） */
#define RVVFFT_VL_F32  (RVVFFT_VLEN / 32)
/* 每个向量寄存器可容纳的f64元素数（LMUL=1） */
#define RVVFFT_VL_F64  (RVVFFT_VLEN / 64)

/* ── 错误码 ───────────────────────────────────────────────── */
typedef enum {
    RVVFFT_SUCCESS          =  0,
    RVVFFT_ERR_INVALID_ARG  = -1, /**< 无效参数（n不是2的幂等） */
    RVVFFT_ERR_NULL_PTR     = -2, /**< 空指针 */
    RVVFFT_ERR_NO_MEMORY    = -3, /**< 内存分配失败 */
    RVVFFT_ERR_NO_PLAN      = -4, /**< 找不到可用plan */
    RVVFFT_ERR_IO           = -5  /**< Wisdom文件I/O错误 */
} rvvfft_status_t;

/* ── 内部codelet函数指针原型 ──────────────────────────────── */
/**
 * @brief 单个FFT codelet的函数签名（标准RVV路径）。
 *
 * @param ri     实部输入（split布局）或交错复数输入起始指针
 * @param ii     虚部输入（split布局；interleaved时为NULL）
 * @param ro     实部输出
 * @param io     虚部输出（split布局；interleaved时为NULL）
 * @param tw     Twiddle factor表指针（预计算，VLEN特化对齐）
 * @param stride 输入/输出步长（以复数元素为单位）
 * @param n      本级处理的DFT长度
 * @param sign   RVVFFT_FORWARD 或 RVVFFT_BACKWARD
 */
typedef void (*rvvfft_codelet_fn)(
    const float *ri, const float *ii,
    float       *ro, float       *io,
    const float *tw,
    ptrdiff_t    stride,
    int          n,
    int          sign
);

/** 双精度版本 */
typedef void (*rvvfft_codelet_fn_f64)(
    const double *ri, const double *ii,
    double       *ro, double       *io,
    const double *tw,
    ptrdiff_t     stride,
    int           n,
    int           sign
);

/* ── 辅助宏 ───────────────────────────────────────────────── */
/** 检查n是否为2的幂（n > 0） */
#define RVVFFT_IS_POW2(n)  ((n) > 0 && (((n) & ((n)-1)) == 0))

/** 向上对齐到align字节 */
#define RVVFFT_ALIGN_UP(x, align) \
    (((uintptr_t)(x) + (align) - 1) & ~((uintptr_t)(align) - 1))

#endif /* RVVFFT_TYPES_H */
