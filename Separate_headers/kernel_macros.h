#pragma once

#define PRAM_THRESHOLD 1e3

#define kernel(blocks, threads) <<<blocks, threads>>>
#define n_threads(thread_count) kernel(thread_count / 32 + (thread_count % 32 > 0), 32)