
#include "data_type.h"
#include "NN_enums.h"
#include "Adam.h"

#pragma once
__device__ IOptimizer* initialize_optimizer(optimizers_enum optimizer);
__global__ void global_optimizer_init(optimizers_enum optimizer, IOptimizer **out);
__host__ IOptimizer* host_optimizer_init(optimizers_enum optimizer);
