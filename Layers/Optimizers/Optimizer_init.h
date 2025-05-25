
#include "data_type.h"
#include "kernel_macros.h"
#include "NN_enums.h"
#include "Adam.h"

#pragma once
__device__ IOptimizer* initialize_optimizer(optimizers_enum optimizer, size_t parameter_count);
__global__ void global_optimizer_init(optimizers_enum optimizer, IOptimizer **out, size_t parameter_count);
__host__ IOptimizer* host_optimizer_init(optimizers_enum optimizer, size_t parameter_count);
__global__ void call_Optimizer_destructor(IOptimizer *optimizer);
__global__ void call_optimizer_values_alloc(IOptimizer* optimizer, size_t new_param_count, bool copy_old_values = false);

// SAVING - LOADING

__global__ void get_optimizer_data_buffer(IOptimizer* optimizer, char* out_buffer, size_t out_buffer_size, size_t* buff_len);
__host__ void host_save_optimizer(FILE *file, IOptimizer *optimizer);
__host__ IOptimizer* host_load_optimizer(FILE* file);
