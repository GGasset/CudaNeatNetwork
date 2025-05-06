#ifndef NEURON_OPERATIONS_H
#define NEURON_OPERATIONS_H

#include "data_type.h"
#include "NN_enums.h"

#include "cuda_functionality.cuh"

__global__ void sigmoid_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values,
	size_t layer_length
);

__global__ void tanh_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values,
	size_t layer_length
);

__global__ void LSTM_execution(
	data_t* activations, size_t activations_start, size_t layer_activations_start,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights, data_t* state,
	size_t layer_length
);
#endif
