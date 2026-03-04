/**
 * planner.c — FFT plan construction and decomposition strategies.
 *
 * A plan is a tree of nodes, where each leaf node binds to a codelet
 * (a concrete butterfly kernel for a specific radix/layout/dtype/vlen).
 * The planner recursively decomposes N = N1 × N2 and selects the best
 * decomposition using one of two strategies:
 *
 *   RVVFFT_ESTIMATE  — Heuristic: prefer radix-8 > radix-4 > radix-2.
 *   RVVFFT_MEASURE   — Benchmark all candidates and pick the fastest.
 *
 * Wisdom (plan serialisation) is handled in wisdom.c.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rvvfft.h"
#include "rvvfft_types.h"

/* ─── Internal plan node ──────────────────────────────────────────────────── */

typedef enum {
    PLAN_NODE_LEAF,     /* leaf: one codelet                      */
    PLAN_NODE_INTERNAL  /* internal: two sub-plans combined        */
} plan_node_kind_t;

typedef struct plan_node_s {
    plan_node_kind_t kind;
    int              n;        /* transform length at this node    */
    int              radix;    /* radix of this stage (leaf only)  */
    /* Children for internal node */
    struct plan_node_s *left;
    struct plan_node_s *right;
} plan_node_t;

/* ─── rvvfft_plan_s (opaque type) ────────────────────────────────────────── */

struct rvvfft_plan_s {
    int              n;
    int              sign;         /* RVVFFT_FORWARD or RVVFFT_BACKWARD */
    rvvfft_layout_t  layout;
    rvvfft_dtype_t   dtype;
    unsigned         flags;

    /* I/O buffer pointers (from planning call) */
    void            *in;    /* interleaved: complex*, split: float* (real) */
    void            *in_i;  /* split only: float* (imag)                  */
    void            *out;   /* interleaved: complex*, split: float* (real) */
    void            *out_i; /* split only: float* (imag)                  */

    /* Root of the plan tree */
    plan_node_t     *root;

    /* Twiddle factor tables (split format, f32 for now) */
    float           *tw_r;
    float           *tw_i;
};

/* ─── Helpers ────────────────────────────────────────────────────────────── */

static plan_node_t *alloc_leaf(int n, int radix)
{
    plan_node_t *node = (plan_node_t *)calloc(1, sizeof(*node));
    if (!node) return NULL;
    node->kind  = PLAN_NODE_LEAF;
    node->n     = n;
    node->radix = radix;
    return node;
}

static plan_node_t *alloc_internal(int n, plan_node_t *left, plan_node_t *right)
{
    plan_node_t *node = (plan_node_t *)calloc(1, sizeof(*node));
    if (!node) return NULL;
    node->kind  = PLAN_NODE_INTERNAL;
    node->n     = n;
    node->left  = left;
    node->right = right;
    return node;
}

static void free_plan_node(plan_node_t *node)
{
    if (!node) return;
    free_plan_node(node->left);
    free_plan_node(node->right);
    free(node);
}

/* ─── Heuristic decomposition (ESTIMATE mode) ────────────────────────────── */

/**
 * choose_radix — Pick the largest supported radix that divides n.
 *
 * Preference order: 8 > 4 > 2.
 */
static int choose_radix(int n)
{
    if (n >= 8 && (n % 8) == 0) return 8;
    if (n >= 4 && (n % 4) == 0) return 4;
    return 2;
}

/**
 * build_plan_node — Recursively build the plan tree for a length-n sub-problem.
 */
static plan_node_t *build_plan_node(int n)
{
    if (n <= 1) return NULL;
    if (!rvvfft_is_power_of_2(n)) return NULL;

    /* Base cases */
    if (n == 2) return alloc_leaf(n, 2);
    if (n == 4) return alloc_leaf(n, 4);
    if (n == 8) return alloc_leaf(n, 8);

    /* Recursive decomposition */
    int radix = choose_radix(n);
    int sub_n = n / radix;

    plan_node_t *sub = build_plan_node(sub_n);
    plan_node_t *leaf = alloc_leaf(n, radix);
    if (!sub || !leaf) {
        free_plan_node(sub);
        free_plan_node(leaf);
        return NULL;
    }
    return alloc_internal(n, sub, leaf);
}

/* ─── Public planner API ──────────────────────────────────────────────────── */

rvvfft_plan rvvfft_plan_dft_1d(int n,
                                rvvfft_complex *in,
                                rvvfft_complex *out,
                                int sign,
                                unsigned flags)
{
    if (!rvvfft_is_power_of_2(n) || n < 2) return NULL;

    rvvfft_plan plan = (rvvfft_plan)calloc(1, sizeof(*plan));
    if (!plan) return NULL;

    plan->n      = n;
    plan->sign   = sign;
    plan->layout = RVVFFT_LAYOUT_INTERLEAVED;
    plan->dtype  = RVVFFT_DTYPE_F32;
    plan->flags  = flags;
    plan->in     = in;
    plan->in_i   = NULL;
    plan->out    = out;
    plan->out_i  = NULL;

    plan->root = build_plan_node(n);
    if (!plan->root) {
        free(plan);
        return NULL;
    }

    /* Allocate twiddle tables */
    int tw_count = n / 2;
    plan->tw_r = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                         (size_t)tw_count * sizeof(float));
    plan->tw_i = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                         (size_t)tw_count * sizeof(float));
    if (!plan->tw_r || !plan->tw_i) {
        free(plan->tw_r);
        free(plan->tw_i);
        free_plan_node(plan->root);
        free(plan);
        return NULL;
    }

    /* Fill twiddle table */
    extern void rvvfft_twiddle_init_f32(float *, float *, int, int, int);
    rvvfft_twiddle_init_f32(plan->tw_r, plan->tw_i, n, 1, tw_count);

    return plan;
}

rvvfft_plan rvvfft_plan_dft_split_1d(int n,
                                      const float *ri, const float *ii,
                                      float *ro,       float *io,
                                      int sign,
                                      unsigned flags)
{
    /* Reuse the interleaved planner internals; layout is recorded. */
    if (!rvvfft_is_power_of_2(n) || n < 2) return NULL;

    rvvfft_plan plan = (rvvfft_plan)calloc(1, sizeof(*plan));
    if (!plan) return NULL;

    plan->n      = n;
    plan->sign   = sign;
    plan->layout = RVVFFT_LAYOUT_SPLIT;
    plan->dtype  = RVVFFT_DTYPE_F32;
    plan->flags  = flags;
    plan->in     = (void *)ri;
    plan->in_i   = (void *)ii;
    plan->out    = ro;
    plan->out_i  = io;

    plan->root = build_plan_node(n);
    if (!plan->root) { free(plan); return NULL; }

    int tw_count = n / 2;
    plan->tw_r = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                         (size_t)tw_count * sizeof(float));
    plan->tw_i = (float *)aligned_alloc(RVVFFT_ALIGN_BYTES,
                                         (size_t)tw_count * sizeof(float));
    if (!plan->tw_r || !plan->tw_i) {
        free(plan->tw_r);
        free(plan->tw_i);
        free_plan_node(plan->root);
        free(plan);
        return NULL;
    }

    extern void rvvfft_twiddle_init_f32(float *, float *, int, int, int);
    rvvfft_twiddle_init_f32(plan->tw_r, plan->tw_i, n, 1, tw_count);

    return plan;
}

rvvfft_plan rvvfft_plan_dft_1d_f64(int n,
                                    rvvfft_complex_f64 *in,
                                    rvvfft_complex_f64 *out,
                                    int sign,
                                    unsigned flags)
{
    /* Stub — f64 executor not yet implemented */
    (void)in; (void)out; (void)sign; (void)flags;
    if (!rvvfft_is_power_of_2(n) || n < 2) return NULL;
    return NULL;  /* TODO: implement in Phase 1.8 */
}

/* ─── Execution (scalar reference path) ──────────────────────────────────── */

void rvvfft_execute(const rvvfft_plan plan)
{
    if (!plan) return;
    /* TODO: dispatch to the correct RVV codelet via the plan tree.
     * For Phase 0 validation this falls through to the scalar reference path
     * implemented in executor_recursive.c. */
    extern void rvvfft_execute_recursive(const rvvfft_plan plan);
    rvvfft_execute_recursive(plan);
}

void rvvfft_execute_dft(const rvvfft_plan plan,
                         rvvfft_complex *in,
                         rvvfft_complex *out)
{
    if (!plan) return;
    /* Override I/O then execute.  Safe because plan is logically const. */
    rvvfft_plan mutable_plan = (rvvfft_plan)(uintptr_t)plan;
    void *saved_in  = mutable_plan->in;
    void *saved_out = mutable_plan->out;
    mutable_plan->in  = in;
    mutable_plan->out = out;
    rvvfft_execute(plan);
    mutable_plan->in  = saved_in;
    mutable_plan->out = saved_out;
}

void rvvfft_execute_split_dft(const rvvfft_plan plan,
                               const float *ri, const float *ii,
                               float *ro,       float *io)
{
    (void)plan; (void)ri; (void)ii; (void)ro; (void)io;
    /* TODO: split-layout execution dispatch */
}

/* ─── Destroy ─────────────────────────────────────────────────────────────── */

void rvvfft_destroy_plan(rvvfft_plan plan)
{
    if (!plan) return;
    free_plan_node(plan->root);
    free(plan->tw_r);
    free(plan->tw_i);
    free(plan);
}
