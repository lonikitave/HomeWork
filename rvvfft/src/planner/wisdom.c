/**
 * wisdom.c — Plan wisdom序列化与反序列化
 *
 * Wisdom机制：将MEASURE模式下实测得到的最优plan信息保存到文件，
 * 下次启动时直接加载，避免重复计时开销。
 *
 * 文件格式（文本，方便调试）：
 * ```
 * # rvvfft wisdom v1 VLEN=256
 * n=1024 sign=-1 layout=split dtype=f32 radix_sequence=8,8,4 time_us=12.3
 * n=4096 sign=-1 layout=split dtype=f32 radix_sequence=8,8,8,8 time_us=55.7
 * ...
 * ```
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/rvvfft.h"
#include "../../include/rvvfft_types.h"

#define WISDOM_VERSION 1
#define WISDOM_MAX_ENTRIES 4096
#define WISDOM_LINE_MAX    256

/* ── Wisdom条目 ─────────────────────────────────────────── */
typedef struct {
    int             n;
    int             sign;
    rvvfft_layout_t layout;
    rvvfft_dtype_t  dtype;
    int             vlen;
    char            radix_seq[64];  /**< 如 "8,8,4" */
    double          time_us;        /**< 实测时间（微秒） */
} wisdom_entry_t;

static wisdom_entry_t g_wisdom[WISDOM_MAX_ENTRIES];
static int g_wisdom_count = 0;

/**
 * @brief 查找与给定参数匹配的wisdom条目。
 * @return 条目指针，未找到返回NULL
 */
const wisdom_entry_t *wisdom_lookup(int n, int sign,
                                    rvvfft_layout_t layout,
                                    rvvfft_dtype_t dtype, int vlen)
{
    for (int i = 0; i < g_wisdom_count; ++i) {
        wisdom_entry_t *e = &g_wisdom[i];
        if (e->n == n && e->sign == sign && e->layout == layout &&
            e->dtype == dtype && e->vlen == vlen)
            return e;
    }
    return NULL;
}

/**
 * @brief 将wisdom条目写入全局缓存（MEASURE模式调用）。
 */
void wisdom_store(int n, int sign, rvvfft_layout_t layout,
                  rvvfft_dtype_t dtype, int vlen,
                  const char *radix_seq, double time_us)
{
    /* 更新已有条目 */
    for (int i = 0; i < g_wisdom_count; ++i) {
        wisdom_entry_t *e = &g_wisdom[i];
        if (e->n == n && e->sign == sign && e->layout == layout &&
            e->dtype == dtype && e->vlen == vlen) {
            if (time_us < e->time_us) {
                strncpy(e->radix_seq, radix_seq, sizeof(e->radix_seq) - 1);
                e->time_us = time_us;
            }
            return;
        }
    }
    /* 新条目 */
    if (g_wisdom_count < WISDOM_MAX_ENTRIES) {
        wisdom_entry_t *e = &g_wisdom[g_wisdom_count++];
        e->n      = n;
        e->sign   = sign;
        e->layout = layout;
        e->dtype  = dtype;
        e->vlen   = vlen;
        strncpy(e->radix_seq, radix_seq, sizeof(e->radix_seq) - 1);
        e->time_us = time_us;
    }
}

/**
 * @brief 将当前wisdom写入文件。
 */
rvvfft_status_t rvvfft_export_wisdom_to_file(const char *filename)
{
    if (!filename) return RVVFFT_ERR_INVALID_ARG;

    FILE *fp = fopen(filename, "w");
    if (!fp) return RVVFFT_ERR_IO;

    fprintf(fp, "# rvvfft wisdom v%d VLEN=%d\n",
            WISDOM_VERSION, rvvfft_get_vlen());
    for (int i = 0; i < g_wisdom_count; ++i) {
        const wisdom_entry_t *e = &g_wisdom[i];
        fprintf(fp, "n=%d sign=%d layout=%d dtype=%d vlen=%d "
                    "radix_seq=%s time_us=%.3f\n",
                e->n, e->sign, (int)e->layout, (int)e->dtype, e->vlen,
                e->radix_seq, e->time_us);
    }
    fclose(fp);
    return RVVFFT_SUCCESS;
}

/**
 * @brief 从文件加载wisdom。
 */
rvvfft_status_t rvvfft_import_wisdom_from_file(const char *filename)
{
    if (!filename) return RVVFFT_ERR_INVALID_ARG;

    FILE *fp = fopen(filename, "r");
    if (!fp) return RVVFFT_ERR_IO;

    char line[WISDOM_LINE_MAX];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;  /* 注释行 */

        wisdom_entry_t e;
        char radix_seq[64] = {0};
        int layout_int = 0, dtype_int = 0;
        int matched = sscanf(line,
            "n=%d sign=%d layout=%d dtype=%d vlen=%d "
            "radix_seq=%63s time_us=%lf",
            &e.n, &e.sign, &layout_int, &dtype_int, &e.vlen,
            radix_seq, &e.time_us);
        if (matched == 7) {
            e.layout = (rvvfft_layout_t)layout_int;
            e.dtype  = (rvvfft_dtype_t)dtype_int;
            e.radix_seq[sizeof(e.radix_seq) - 1] = '\0';
            memcpy(e.radix_seq, radix_seq, sizeof(e.radix_seq) - 1);
            wisdom_store(e.n, e.sign, e.layout, e.dtype, e.vlen,
                         e.radix_seq, e.time_us);
        }
    }
    fclose(fp);
    return RVVFFT_SUCCESS;
}

/**
 * @brief 清除所有缓存的wisdom。
 */
void rvvfft_forget_wisdom(void)
{
    memset(g_wisdom, 0, sizeof(g_wisdom));
    g_wisdom_count = 0;
}
