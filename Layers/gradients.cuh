#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"
#include "cuda_functionality.cuh"
#include "neuron_operations.cuh"
#include "derivatives.cuh"


__global__ void LSTM_gradient_calculation(
	data_t* derivatives, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* gradients, size_t gradients_start, size_t next_t_gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	data_t* costs, size_t costs_start, size_t layer_costs_start,
	size_t layer_length
);

__global__ void LSTM_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	field_t* neuron_weights,
	data_t learning_rate, data_t max_subtracted_gradient,
	size_t layer_length
);

__global__ void neuron_gradient_calculation(
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start, size_t layer_costs_start,
	ActivationFunctions activation,
	size_t layer_length
);

__global__ void cud_set_dropout(
	float dropout_rate, float* normalized_random_samples, short* dropout,
	size_t layer_length
);