#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"
#include "cuda_functionality.cuh"

__device__ data_t device_sigmoid_derivative(
	data_t input
);

__device__ data_t device_tanh_derivative(
	data_t input
);

__global__ void LSTM_derivative_calculation(
	data_t* prev_state_derivatives, data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights,
	size_t layer_length
);