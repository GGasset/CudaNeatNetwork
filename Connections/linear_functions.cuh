#pragma once

#include "data_type.h"
#include "cuda_functionality.cuh"

__global__ void cud_add_biases(
	size_t layer_length, field_t* biases,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values);

__global__ void cud_dense_linear_function(
	size_t previous_layer_length, field_t* weights,
	size_t activations_start, size_t previous_layer_activations_start, data_t* activations,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values
);

__global__ void cud_NEAT_linear_function(
	size_t connection_count, field_t* weights, size_t* connection_points, size_t* connection_neuron_i,
	size_t activations_start, data_t* activations,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values
);

__global__ void cud_dense_linear_function_derivative(
	size_t activations_start, size_t previous_layer_activations_start, size_t previous_layer_length, data_t* activations,
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives,
	field_t* weights
);

__global__ void cud_NEAT_linear_function_derivative(
	size_t activations_start, data_t* activations,
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives,
	size_t connection_count, field_t* weights, size_t* connection_points, size_t* connection_neuron_i
);

__global__ void NEAT_unsummed_linear_func_derivative(
	size_t t_count, size_t t_count_gap, size_t neuron_count,
	size_t *connection_points, size_t *connection_neuron_i, size_t layer_max_connection_count, size_t connection_count,
	nn_lens lens, layer_properties props, data_t *weights, data_t *activations, data_t *out_arr
);

__global__ void cud_add_bias_derivative(
	size_t layer_length,
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
);
