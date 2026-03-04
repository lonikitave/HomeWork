/**
 * rvvfft.h — 公共API主头文件
 *
 * RISC-V向量(RVV)高性能FFT库主接口，参考FFTW的plan/execute分层设计。
 * 支持交错(interleaved)和分离(split)双数据布局、1D C2C与R2C变换。
 */
#ifndef RVVFFT_H
#define RVVFFT_H

#include "rvvfft_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Plan句柄 ────────────────────────────────────────────── */
/** 不透明plan对象，由planner创建，传给executor执行 */
typedef struct rvvfft_plan_s *rvvfft_plan;

/* ═══════════════════════════════════════════════════════════
 * Plan创建 API
 * ═══════════════════════════════════════════════════════════ */

/**
 * @brief 为1D复数FFT（交错布局）创建plan。
 *
 * @param n     变换长度（必须是2的幂，Phase 1仅支持2^k, 1≤k≤24）
 * @param in    输入缓冲区（rvvfft_complex[]，长度≥n）
 * @param out   输出缓冲区（可与in相同实现in-place）
 * @param sign  RVVFFT_FORWARD(-1) 或 RVVFFT_BACKWARD(+1)
 * @param flags RVVFFT_ESTIMATE | RVVFFT_MEASURE | …
 * @return      plan句柄，失败返回NULL
 */
rvvfft_plan rvvfft_plan_dft_1d(
    int             n,
    rvvfft_complex *in,
    rvvfft_complex *out,
    int             sign,
    unsigned        flags
);

/**
 * @brief 为1D复数FFT（分离布局）创建plan。
 *
 * @param n   变换长度
 * @param ri  输入实部数组（float[]，长度≥n）
 * @param ii  输入虚部数组（float[]，长度≥n）
 * @param ro  输出实部数组
 * @param io  输出虚部数组
 * @param sign  FFT方向
 * @param flags Plan标志
 * @return    plan句柄，失败返回NULL
 */
rvvfft_plan rvvfft_plan_dft_split_1d(
    int         n,
    const float *ri, const float *ii,
    float       *ro, float       *io,
    int          sign,
    unsigned     flags
);

/**
 * @brief 为1D实数→复数FFT创建plan（R2C）。
 *
 * 利用共轭对称性，输出长度为n/2+1个复数。
 *
 * @param n   输入实数点数（必须是2的幂）
 * @param in  实数输入数组（float[]，长度≥n）
 * @param out 复数输出数组（rvvfft_complex[]，长度≥n/2+1）
 * @param flags Plan标志
 * @return    plan句柄，失败返回NULL
 */
rvvfft_plan rvvfft_plan_dft_r2c_1d(
    int             n,
    const float    *in,
    rvvfft_complex *out,
    unsigned        flags
);

/**
 * @brief 为1D复数→实数FFT创建plan（C2R）。
 *
 * @param n   输出实数点数（必须是2的幂）
 * @param in  复数输入数组（rvvfft_complex[]，长度≥n/2+1）
 * @param out 实数输出数组（float[]，长度≥n）
 * @param flags Plan标志
 * @return    plan句柄，失败返回NULL
 */
rvvfft_plan rvvfft_plan_dft_c2r_1d(
    int                   n,
    const rvvfft_complex *in,
    float                *out,
    unsigned              flags
);

/* ═══════════════════════════════════════════════════════════
 * Executor API
 * ═══════════════════════════════════════════════════════════ */

/**
 * @brief 执行已创建的FFT plan。
 *
 * 线程安全：同一plan可被并发调用（只读plan树），
 * 但各自的输入/输出缓冲区必须独立。
 *
 * @param plan 由rvvfft_plan_*创建的plan句柄
 */
void rvvfft_execute(const rvvfft_plan plan);

/**
 * @brief 用新缓冲区执行plan（避免重新创建plan的开销）。
 *
 * 新缓冲区的对齐和大小必须与创建plan时相同。
 *
 * @param plan  已有plan
 * @param in    新输入缓冲区（交错布局）
 * @param out   新输出缓冲区
 */
void rvvfft_execute_dft(
    const rvvfft_plan plan,
    rvvfft_complex   *in,
    rvvfft_complex   *out
);

/**
 * @brief 用新缓冲区执行split布局plan。
 */
void rvvfft_execute_dft_split(
    const rvvfft_plan plan,
    const float *ri, const float *ii,
    float       *ro, float       *io
);

/* ═══════════════════════════════════════════════════════════
 * Plan销毁 & Wisdom
 * ═══════════════════════════════════════════════════════════ */

/**
 * @brief 销毁plan并释放资源。
 * @param plan 待销毁plan（NULL时无操作）
 */
void rvvfft_destroy_plan(rvvfft_plan plan);

/**
 * @brief 将当前wisdom（已测量的最优plan信息）写入文件。
 * @param filename 输出文件路径
 * @return RVVFFT_SUCCESS 或错误码
 */
rvvfft_status_t rvvfft_export_wisdom_to_file(const char *filename);

/**
 * @brief 从文件加载wisdom，供后续plan创建使用。
 * @param filename wisdom文件路径
 * @return RVVFFT_SUCCESS 或错误码
 */
rvvfft_status_t rvvfft_import_wisdom_from_file(const char *filename);

/** 清除所有缓存的wisdom。 */
void rvvfft_forget_wisdom(void);

/* ═══════════════════════════════════════════════════════════
 * 库初始化 & 版本信息
 * ═══════════════════════════════════════════════════════════ */

/**
 * @brief 初始化库：检测运行时VLEN，预计算twiddle表等。
 *
 * 在首次调用rvvfft_plan_*之前调用（也可不调用，首次plan时自动初始化）。
 *
 * @param vlen 期望的VLEN（RVVFFT_VLEN_AUTO=0则自动检测）
 */
void rvvfft_init(rvvfft_vlen_t vlen);

/** 清理全局资源（如twiddle表内存） */
void rvvfft_cleanup(void);

/* ═══════════════════════════════════════════════════════════
 * 内存工具
 * ═══════════════════════════════════════════════════════════ */

/**
 * @brief 分配按RVVFFT_ALIGN_BYTES对齐的内存。
 * @param size 分配字节数
 * @return 对齐指针，失败返回NULL
 */
void *rvvfft_aligned_malloc(size_t size);

/**
 * @brief 分配并清零对齐内存。
 */
void *rvvfft_aligned_calloc(size_t count, size_t size);

/**
 * @brief 释放由rvvfft_aligned_malloc分配的内存。
 */
void rvvfft_aligned_free(void *ptr);

/**
 * @brief 检查指针是否按RVVFFT_ALIGN_BYTES对齐。
 * @return 1对齐，0未对齐
 */
int rvvfft_is_aligned(const void *ptr);

/** 返回库版本字符串，如 "0.1.0-VLEN256" */
const char *rvvfft_version(void);

/** 返回当前使用的VLEN（位） */
int rvvfft_get_vlen(void);

#ifdef __cplusplus
}
#endif

#endif /* RVVFFT_H */
