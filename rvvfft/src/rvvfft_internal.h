/**
 * rvvfft_internal.h — 内部共享结构体定义
 *
 * 供planner.c和executor_recursive.c共享plan数据结构。
 * 此头文件不属于公共API，外部用户不应包含它。
 */
#ifndef RVVFFT_INTERNAL_H
#define RVVFFT_INTERNAL_H

#include "../include/rvvfft_types.h"

/* ── Plan节点类型 ─────────────────────────────────────────── */
typedef enum {
    PLAN_NODE_LEAF     = 0,
    PLAN_NODE_COOLEY   = 1,
    PLAN_NODE_REAL_HALF = 2
} plan_node_type_t;

/* ── Codelet描述符（前向声明，完整定义在planner.c） ────────── */
typedef struct codelet_desc_s {
    int               radix;
    rvvfft_layout_t   layout;
    rvvfft_dtype_t    dtype;
    int               vlen;
    rvvfft_codelet_fn fn;
    const char       *name;
} codelet_desc_t;

/* ── Plan树节点 ────────────────────────────────────────────── */
struct rvvfft_plan_node_s {
    plan_node_type_t type;
    int              n;
    int              sign;
    rvvfft_layout_t  layout;

    union {
        struct {
            const codelet_desc_t    *codelet;
            const float             *tw_real;
            const float             *tw_imag;
        } leaf;
        struct {
            int                      n1;
            int                      n2;
            struct rvvfft_plan_node_s *child_dft_n2;
            struct rvvfft_plan_node_s *child_dft_n1;
        } cooley;
    } u;
};
typedef struct rvvfft_plan_node_s plan_node_t;

/* ── Plan句柄完整定义 ──────────────────────────────────────── */
struct rvvfft_plan_s {
    int             n;
    int             sign;
    rvvfft_layout_t layout;
    rvvfft_dtype_t  dtype;
    unsigned        flags;
    plan_node_t    *root;
    int            *bitrev_perm;
    double          measured_time;
};

#endif /* RVVFFT_INTERNAL_H */
