
#pragma once

#include "data_type.h"
#include "NN_enums.h"

#include "cuda_functionality.cuh"
#include "nn_lens.h"

__device__ data_t sigmoid_activation(data_t in);
__device__ data_t tanh_activation(data_t in);
__device__ data_t sofmax_activation(data_t in, data_t exponent_sum);

__global__ void g_activation_function(
	size_t t_count, data_t *execution_vals, data_t *activations,
	ActivationFunctions activation, layer_properties, nn_lens, size_t timestep_gap 
);
__host__ void activation_function(
	size_t t_count, data_t *execution_vals, data_t *activations,
	ActivationFunctions activation, layer_properties, nn_lens, size_t timestep_gap 
);

__device__ data_t sigmoid_derivative(data_t in);
__device__ data_t tanh_derivative(data_t in);
__device__ data_t sofmax_derivative(data_t in, data_t exponent_sum);
__global__ void backpropagate_activation(
	size_t t_count, data_t *execution_vals, data_t *gradients, data_t *costs, 
	ActivationFunctions activation, layer_properties, nn_lens, size_t timestep_gap 
);


