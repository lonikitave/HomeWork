/**
 * planner.c — FFT Plan构建与策略选择
 *
 * Planner是rvvfft的核心调度器，参考FFTW的codelet+planner架构：
 *
 *   1. 接受用户输入（n, 布局, 精度, flags）
 *   2. 将N分解为 N = radix^k 或 N = N1×N2 的递归形式
 *   3. 为每个子问题选择最优codelet（radix-2/4/8 × split/interleaved × VLEN特化）
 *   4. 构建Plan树（每个节点代表一个FFT pass）
 *   5. ESTIMATE模式：启发式选择（优先radix-8 → radix-4 → radix-2）
 *      MEASURE模式：实际计时所有候选plan，选最快的
 *   6. 查询wisdom缓存，避免重复测量
 *
 * Plan树节点结构：
 *   - 叶子节点：绑定具体codelet函数指针
 *   - 内部节点：描述递归分解（N = N1×N2）和twiddle旋转
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../../include/rvvfft.h"
#include "../../include/rvvfft_types.h"
#include "../rvvfft_internal.h"

/* ── 全局状态 ─────────────────────────────────────────────── */
static int g_vlen = RVVFFT_VLEN;  /* 运行时VLEN，由rvvfft_init设置 */

/* ══════════════════════════════════════════════════════════
 * Codelet注册表（前向声明外部kernel函数）
 * ══════════════════════════════════════════════════════════ */

/* radix-2 标量参考 */
extern void rvvfft_r2_split_f32_scalar(
    float *real, float *imag, const float *tw_r, const float *tw_i,
    int n, ptrdiff_t stride, int sign);
extern void rvvfft_r2_interleaved_f32_scalar(
    rvvfft_complex_f32 *data, const float *tw_r, const float *tw_i,
    int n, ptrdiff_t stride, int sign);

/* radix-4 标量参考 */
extern void rvvfft_r4_split_f32_scalar(
    float *real, float *imag,
    const float *tw1r, const float *tw1i,
    const float *tw2r, const float *tw2i,
    const float *tw3r, const float *tw3i,
    int n, ptrdiff_t stride, int sign);

/* radix-8 标量参考 */
extern void rvvfft_r8_split_f32_scalar(
    float *real, float *imag, const float *tw,
    int n, ptrdiff_t stride, int sign);

/*
 * 统一codelet适配器：将各radix的不同参数签名包装为统一的rvvfft_codelet_fn。
 * Planner通过此适配器调用各radix kernel，隔离参数差异。
 */
static void codelet_r2_split_adapter(
    const float *ri, const float *ii, float *ro, float *io,
    const float *tw, ptrdiff_t stride, int n, int sign)
{
    /* split布局：in-place，ri/ii指向同一缓冲区的real/imag部分 */
    rvvfft_r2_split_f32_scalar((float *)ri, (float *)ii, tw, tw + n/2,
                                n, stride, sign);
    /* 输出已写入ro/io（in-place场景下ro==ri, io==ii） */
    (void)ro; (void)io;
}

static void codelet_r4_split_adapter(
    const float *ri, const float *ii, float *ro, float *io,
    const float *tw, ptrdiff_t stride, int n, int sign)
{
    int quarter = n / 4;
    rvvfft_r4_split_f32_scalar(
        (float *)ri, (float *)ii,
        tw,              tw +   quarter,  /* W^k */
        tw + 2*quarter,  tw + 3*quarter,  /* W^{2k} */
        tw + 4*quarter,  tw + 5*quarter,  /* W^{3k} */
        n, stride, sign);
    (void)ro; (void)io;
}

static void codelet_r8_split_adapter(
    const float *ri, const float *ii, float *ro, float *io,
    const float *tw, ptrdiff_t stride, int n, int sign)
{
    rvvfft_r8_split_f32_scalar((float *)ri, (float *)ii,
                                tw, n, stride, sign);
    (void)ro; (void)io;
}

/**
 * @brief 全局codelet注册表。
 *
 * Planner从此表中按(radix, layout, dtype, vlen)匹配最优codelet。
 * Phase 2代码生成器会自动扩充此表（~60+条目）。
 */
static const codelet_desc_t g_codelet_registry[] = {
    /* radix-2, split, f32, 通用（标量参考） */
    { 2, RVVFFT_LAYOUT_SPLIT, RVVFFT_DTYPE_F32, 0,
      codelet_r2_split_adapter, "r2_split_f32_scalar" },
    /* radix-4, split, f32, 通用 */
    { 4, RVVFFT_LAYOUT_SPLIT, RVVFFT_DTYPE_F32, 0,
      codelet_r4_split_adapter, "r4_split_f32_scalar" },
    /* radix-8, split, f32, 通用 */
    { 8, RVVFFT_LAYOUT_SPLIT, RVVFFT_DTYPE_F32, 0,
      codelet_r8_split_adapter, "r8_split_f32_scalar" },
    /* 哨兵 */
    { 0, 0, 0, 0, NULL, NULL }
};

/* ══════════════════════════════════════════════════════════
 * Planner内部函数
 * ══════════════════════════════════════════════════════════ */

/**
 * @brief 从注册表中选取最优codelet。
 *
 * 优先级：radix匹配 → VLEN匹配 → 布局匹配。
 * ESTIMATE模式下优先选radix-8，其次radix-4，最后radix-2。
 */
static const codelet_desc_t *select_codelet(int radix, rvvfft_layout_t layout,
                                             rvvfft_dtype_t dtype, int vlen)
{
    const codelet_desc_t *best = NULL;
    for (int i = 0; g_codelet_registry[i].fn; ++i) {
        const codelet_desc_t *c = &g_codelet_registry[i];
        if (c->radix != radix || c->layout != layout || c->dtype != dtype)
            continue;
        if (c->vlen != 0 && c->vlen != vlen)
            continue;
        /* 优先VLEN精确匹配 */
        if (!best || (c->vlen == vlen && best->vlen != vlen))
            best = c;
    }
    return best;
}

/**
 * @brief 为长度n的FFT选择分解策略（ESTIMATE启发式）。
 *
 * 策略：
 *   1. 若n能被8整除 → 选radix-8（最少pass数）
 *   2. 若n能被4整除 → 选radix-4
 *   3. 否则 → 选radix-2
 *
 * @return 选中的radix，若n=1返回1（基本情况）
 */
static int choose_radix_estimate(int n)
{
    if (n == 1) return 1;
    if (n % 8 == 0) return 8;
    if (n % 4 == 0) return 4;
    return 2;
}

/* 前向声明 */
static plan_node_t *build_plan_node(int n, int sign, rvvfft_layout_t layout,
                                    rvvfft_dtype_t dtype, unsigned flags,
                                    int vlen);
static void free_plan_node(plan_node_t *node);

/**
 * @brief 递归构建Plan树节点。
 *
 * @param n      DFT长度
 * @param sign   FFT方向
 * @param layout 数据布局
 * @param dtype  数据类型
 * @param flags  Plan flags（ESTIMATE/MEASURE）
 * @param vlen   目标VLEN
 * @return       分配的plan节点（调用者负责free_plan_node释放）
 */
static plan_node_t *build_plan_node(int n, int sign, rvvfft_layout_t layout,
                                    rvvfft_dtype_t dtype, unsigned flags,
                                    int vlen)
{
    plan_node_t *node = calloc(1, sizeof(*node));
    if (!node) return NULL;

    node->n      = n;
    node->sign   = sign;
    node->layout = layout;

    /* 基本情况：n=1，无需计算 */
    if (n == 1) {
        node->type = PLAN_NODE_LEAF;
        node->u.leaf.codelet = NULL;  /* 恒等映射 */
        return node;
    }

    /* 选择分解策略 */
    int radix = choose_radix_estimate(n);

    /* 尝试从注册表获取对应radix的codelet */
    const codelet_desc_t *cdl = select_codelet(radix, layout, dtype, vlen);

    if (!cdl) {
        /* 注册表中没有对应codelet，尝试fallback */
        if (radix == 8) radix = 4;
        else if (radix == 4) radix = 2;
        cdl = select_codelet(radix, layout, dtype, vlen);
    }

    if (cdl && (n % radix == 0)) {
        /*
         * 叶子节点：此节点直接用一个codelet处理长度为n的DFT。
         * 对于大n，实际的Cooley-Tukey pass由codelet内部循环完成。
         */
        node->type = PLAN_NODE_LEAF;
        node->u.leaf.codelet = cdl;

        /* 预取twiddle表（通过twiddle.c的缓存机制） */
        extern rvvfft_status_t rvvfft_twiddle_get_f32(
            int, int, const float **, const float **);
        rvvfft_twiddle_get_f32(n, sign,
                               &node->u.leaf.tw_real,
                               &node->u.leaf.tw_imag);
    } else {
        /*
         * 内部Cooley-Tukey节点：N = n1 × n2
         * 递归构建子plan。Phase 3完善后这里会引入MEASURE比较。
         */
        node->type = PLAN_NODE_COOLEY;
        int n1 = radix > 0 ? radix : 2;
        int n2 = n / n1;
        node->u.cooley.n1 = n1;
        node->u.cooley.n2 = n2;

        node->u.cooley.child_dft_n2 = build_plan_node(
            n2, sign, layout, dtype, flags, vlen);
        node->u.cooley.child_dft_n1 = build_plan_node(
            n1, sign, layout, dtype, flags, vlen);

        if (!node->u.cooley.child_dft_n2 || !node->u.cooley.child_dft_n1) {
            free_plan_node(node);
            return NULL;
        }
    }

    return node;
}

/**
 * @brief 递归释放Plan树节点。
 */
static void free_plan_node(plan_node_t *node)
{
    if (!node) return;
    if (node->type == PLAN_NODE_COOLEY) {
        free_plan_node(node->u.cooley.child_dft_n2);
        free_plan_node(node->u.cooley.child_dft_n1);
    }
    free(node);
}

/* ══════════════════════════════════════════════════════════
 * 公共API实现
 * ══════════════════════════════════════════════════════════ */

/**
 * @brief 为1D复数FFT（交错布局）创建plan。
 */
rvvfft_plan rvvfft_plan_dft_1d(int n, rvvfft_complex *in,
                                rvvfft_complex *out,
                                int sign, unsigned flags)
{
    if (!RVVFFT_IS_POW2(n) || !in || !out) return NULL;

    rvvfft_plan plan = calloc(1, sizeof(*plan));
    if (!plan) return NULL;

    plan->n       = n;
    plan->sign    = sign;
    plan->layout  = RVVFFT_LAYOUT_INTERLEAVED;
    plan->dtype   = RVVFFT_DTYPE_F32;
    plan->flags   = flags;
    plan->measured_time = -1.0;

    /*
     * Interleaved布局策略：
     *   - 内部执行时转为split格式（性能更优）
     *   - Plan树按split布局构建
     */
    plan->root = build_plan_node(n, sign, RVVFFT_LAYOUT_SPLIT,
                                 RVVFFT_DTYPE_F32, flags, g_vlen);
    if (!plan->root) {
        free(plan);
        return NULL;
    }

    /* Bit-reversal置换表（DIT模式需要） */
    extern void rvvfft_bitrev_table(int, int *);
    plan->bitrev_perm = malloc((size_t)n * sizeof(int));
    if (plan->bitrev_perm)
        rvvfft_bitrev_table(n, plan->bitrev_perm);

    return plan;
}

/**
 * @brief 为1D复数FFT（分离布局）创建plan。
 */
rvvfft_plan rvvfft_plan_dft_split_1d(int n,
                                      const float *ri, const float *ii,
                                      float *ro, float *io,
                                      int sign, unsigned flags)
{
    if (!RVVFFT_IS_POW2(n) || !ri || !ii || !ro || !io) return NULL;

    rvvfft_plan plan = calloc(1, sizeof(*plan));
    if (!plan) return NULL;

    plan->n      = n;
    plan->sign   = sign;
    plan->layout = RVVFFT_LAYOUT_SPLIT;
    plan->dtype  = RVVFFT_DTYPE_F32;
    plan->flags  = flags;
    plan->measured_time = -1.0;

    plan->root = build_plan_node(n, sign, RVVFFT_LAYOUT_SPLIT,
                                 RVVFFT_DTYPE_F32, flags, g_vlen);
    if (!plan->root) { free(plan); return NULL; }

    extern void rvvfft_bitrev_table(int, int *);
    plan->bitrev_perm = malloc((size_t)n * sizeof(int));
    if (plan->bitrev_perm)
        rvvfft_bitrev_table(n, plan->bitrev_perm);

    return plan;
}

/**
 * @brief 为1D实数→复数FFT创建plan（R2C）。
 */
rvvfft_plan rvvfft_plan_dft_r2c_1d(int n, const float *in,
                                    rvvfft_complex *out, unsigned flags)
{
    if (!RVVFFT_IS_POW2(n) || !in || !out) return NULL;

    /* R2C：利用共轭对称性，对length-n/2的复数FFT后后处理 */
    rvvfft_plan plan = calloc(1, sizeof(*plan));
    if (!plan) return NULL;

    plan->n      = n;
    plan->sign   = RVVFFT_FORWARD;
    plan->layout = RVVFFT_LAYOUT_SPLIT;
    plan->dtype  = RVVFFT_DTYPE_F32;
    plan->flags  = flags;
    plan->measured_time = -1.0;

    /* 内部：对n/2点复数FFT */
    plan->root = build_plan_node(n / 2, RVVFFT_FORWARD, RVVFFT_LAYOUT_SPLIT,
                                 RVVFFT_DTYPE_F32, flags, g_vlen);
    if (!plan->root) { free(plan); return NULL; }

    return plan;
}

/**
 * @brief 为1D复数→实数FFT创建plan（C2R）。
 */
rvvfft_plan rvvfft_plan_dft_c2r_1d(int n, const rvvfft_complex *in,
                                    float *out, unsigned flags)
{
    if (!RVVFFT_IS_POW2(n) || !in || !out) return NULL;

    rvvfft_plan plan = calloc(1, sizeof(*plan));
    if (!plan) return NULL;

    plan->n      = n;
    plan->sign   = RVVFFT_BACKWARD;
    plan->layout = RVVFFT_LAYOUT_SPLIT;
    plan->dtype  = RVVFFT_DTYPE_F32;
    plan->flags  = flags;
    plan->measured_time = -1.0;

    plan->root = build_plan_node(n / 2, RVVFFT_BACKWARD, RVVFFT_LAYOUT_SPLIT,
                                 RVVFFT_DTYPE_F32, flags, g_vlen);
    if (!plan->root) { free(plan); return NULL; }

    return plan;
}

/**
 * @brief 销毁plan并释放资源。
 */
void rvvfft_destroy_plan(rvvfft_plan plan)
{
    if (!plan) return;
    free_plan_node(plan->root);
    free(plan->bitrev_perm);
    free(plan);
}

/* ── 库初始化 ─────────────────────────────────────────────── */

void rvvfft_init(rvvfft_vlen_t vlen)
{
    if (vlen == RVVFFT_VLEN_AUTO) {
        /* 运行时检测：在真实RISC-V硬件上可通过读CSR vlenb来获取 */
#ifdef __riscv_v
        /* vlenb = VLEN/8，通过内联汇编读取 */
        unsigned long vlenb = 0;
        __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
        g_vlen = (int)(vlenb * 8);
#else
        g_vlen = RVVFFT_VLEN; /* 编译时默认值 */
#endif
    } else {
        g_vlen = (int)vlen;
    }
}

void rvvfft_cleanup(void)
{
    extern void rvvfft_twiddle_free_all(void);
    rvvfft_twiddle_free_all();
}

const char *rvvfft_version(void)
{
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d.%d-VLEN%d",
             RVVFFT_VERSION_MAJOR, RVVFFT_VERSION_MINOR, RVVFFT_VERSION_PATCH,
             g_vlen);
    return buf;
}

int rvvfft_get_vlen(void) { return g_vlen; }
