#pragma once
#include "data_type.h"
#include "NN_enums.h"

#include "cuda_functionality.cuh"
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void MSE_derivative(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* costs, size_t costs_start,
	data_t* Y_hat, size_t output_length
);

__global__ void MSE_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* Y_hat, size_t output_length,
	data_t* cost_write
);

__global__ void log_likelyhood_derivative(
	data_t* activations, size_t activations_start,
	size_t neuron_count, size_t last_layer_activations_start, size_t output_length,
	data_t* costs, size_t costs_start,
	data_t* rewards
);

__global__ void log_likelyhood_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* rewards, size_t output_length,
	data_t* cost
);

__global__ void PPO_cost(
	data_t* activations, size_t activations_start,
	size_t neuron_count, size_t last_layer_activations_start, size_t output_length,
	data_t* costs, size_t costs_start,
	data_t* rewards
);
