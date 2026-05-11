#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PRAM_THRESHOLD 1e3

#ifndef block_size
# ifdef DEBUG
#  define block_size 32
# else
#  define block_size 128
# endif
#endif

#define kernel_shared(blocks, threads, shared_byte_size) <<<blocks, threads, shared_byte_size>>>
#define kernel(blocks, threads) <<<blocks, threads>>>

#define n_threads(thread_count) kernel(thread_count / block_size + (thread_count % block_size > 0), block_size)
#define n_threads_shared(thread_count, shared_byte_size) kernel(thread_count / block_size + (thread_count % block_size > 0), block_size, shared_byte_size)