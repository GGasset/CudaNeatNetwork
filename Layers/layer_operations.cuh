
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

// ## Recurrent states
// If the initial cell/hidden state execution value is != 0, that value is used as state
// else, if gap != 0, they are gathered from the previous execution values
// else, they are initialized to 0
// ## Params:
// t_count: number of parallel execution lines
// timestep_gap: number of executions inside each execution line preceding the latest execution, which will be used
__global__ void LSTM_execution(
	size_t t_count, data_t *execution_values, data_t *activations, data_t *weights,
	layer_properties, nn_lens, size_t timestep_gap
);

__device__ data_t sigmoid_derivative(data_t in);
__device__ data_t tanh_derivative(data_t in);
__device__ data_t sofmax_derivative(data_t in, data_t exponent_sum);
__global__ data_t LSTM_derivatives(
	size_t t_count, data_t *activations, data_t *execution_vals, data_t *derivatives, data_t *weights,
	nn_lens lens, layer_properties layer, size_t timestep_gap
);

__global__ void backpropagate_activation(
	size_t t_count, data_t *execution_vals, data_t *gradients, data_t *costs, 
	ActivationFunctions activation, layer_properties, nn_lens, size_t timestep_gap 
);


