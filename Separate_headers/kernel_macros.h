#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PRAM_THRESHOLD 1e3

#define kernel_shared(blocks, threads, shared_byte_size) <<<blocks, threads, shared_byte_size>>>
#define kernel(blocks, threads) <<<blocks, threads>>>

#define n_threads(thread_count) kernel(thread_count / 32 + (thread_count % 32 > 0), 32)
#define n_threads_shared(thread_count, shared_byte_size) kernel(thread_count / 32 + (thread_count % 32 > 0), 32, shared_byte_size)