
#pragma once

#include "data_type.h"
#include "cuda_functionality.cuh"

template<typename T>
__global__ void g_transpose(data_t *in, size_t in_len, size_t n_cols, data_t *out, size_t out_len)
{
    size_t tid = get_tid();

    if (in_len != out_len || in_len % n_cols || tid >= in_len) return;

    size_t n_rows = in_len / n_cols;
    size_t out_col_i = tid / n_rows;
    size_t out_row_i = tid / n_cols;

    size_t out_i = out_row_i * n_cols + out_col_i;

    out[out_i] = in[tid];
}
