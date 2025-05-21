
#include "data_type.h"
#include "NN_enums.h"
#include "Adam.h"

#pragma once
__device__ IOptimizer* initialize_optimizer(optimizers_enum optimizer, size_t parameter_count);
__global__ void global_optimizer_init(optimizers_enum optimizer, IOptimizer **out, size_t parameter_count);
__host__ IOptimizer* host_optimizer_init(optimizers_enum optimizer, size_t parameter_count);
__global__ void call_Optimizer_destructor(IOptimizer *optimizer);


// SAVING - LOADING

__global__ void get_optimizer_data_buffer(IOptimizer *optimizer, void **out_buffer);
__host__ void host_save_optimizer(FILE *file, IOptimizer *optimizer);
