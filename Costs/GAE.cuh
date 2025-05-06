#pragma once
#include "data_type.h"
#include "NN_enums.h"

#include "cuda_functionality.cuh"
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Params:
//	gamma:
//		Discount factor
__global__ void calculate_discounted_rewards(
	size_t t_count,
	data_t gamma,
	data_t *rewards,
	data_t *discounted_rewards
);

__global__ void calculate_deltas(
	size_t t_count,
	data_t gamma,
	data_t *rewards,
	data_t *value_functions,
	data_t *deltas
);

__global__ void parallel_calculate_GAE_advantage(
	size_t t_count,
	data_t gamma,
	data_t lambda,
	data_t *deltas,
	data_t *advantages
);


