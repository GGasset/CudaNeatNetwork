#ifndef CUDA_ACTIVATIONS
#define CUDA_ACTIVATIONS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "neuron_operations.cuh"

#include <cmath>

__device__ data_t device_sigmoid_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values
)
{
	size_t tid = get_tid();
	size_t neuron_execution_values_start = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid;
	data_t activation = 1 / (1 + exp(-execution_values[neuron_execution_values_start + neuron_execution_values_read]));
	if (write_activation)
	{
		size_t activations_i = activations_start + layer_activation_start + tid;
		activations[activations_i] = activation;
	}
	if (write_execution_values)
	{
		size_t execution_values_i = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid + neuron_execution_values_write;
		execution_values[execution_values_i] = activation;
	}
	return activation;
}

__global__ void sigmoid_activation(
	data_t *activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t *execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values,
	size_t layer_length
)
{
	if (get_tid() >= layer_length) return;
	device_sigmoid_activation(
		activations, activations_start,
		layer_activation_start, write_activation,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_execution_values_read, neuron_execution_values_write, write_execution_values
	);
}

__device__ data_t device_tanh_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values
)
{
	size_t tid = get_tid();
	size_t neuron_execution_values_start = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid;

	data_t x = execution_values[neuron_execution_values_start + neuron_execution_values_read];
	data_t exp_x = exp(x);
	data_t exp_minus_x = exp(-x);
	data_t activation = (exp_x - exp_minus_x) / (exp_x + exp_minus_x);
	if (write_activation)
	{
		size_t activations_i = activations_start + layer_activation_start + tid;
		activations[activations_i] = activation;
	}
	if (write_execution_values)
	{
		size_t execution_values_i = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid + neuron_execution_values_write;
		execution_values[execution_values_i] = activation;
	}
	return activation;
}


__global__ void tanh_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values,
	size_t layer_length
)
{
	if (get_tid() >= layer_length) return;
	device_tanh_activation(
		activations, activations_start, layer_activation_start, write_activation,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_execution_values_read, neuron_execution_values_write, write_execution_values
	);
}

__global__ void LSTM_execution(
	data_t* activations, size_t activations_start, size_t layer_activations_start,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights, data_t* state,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t neuron_weights_start = static_cast<size_t>(4) * tid;
	size_t neuron_state_start = static_cast<size_t>(2) * tid;
	data_t cell_state = state[neuron_state_start];

	size_t neuron_execution_values_start = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid;

	execution_values[neuron_execution_values_start + 3] = cell_state;

	execution_values[neuron_execution_values_start] += state[neuron_state_start + 1];

	data_t linear_sigmoid = device_sigmoid_activation(
		0, 0, 0, 0,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		0, 1, 1
	);

	// Forget Gate
	data_t forget_gate_output = execution_values[neuron_execution_values_start + 2] = neuron_weights[neuron_weights_start] * linear_sigmoid;
	cell_state *= forget_gate_output;

	// Store Gate
	data_t store_gate_sigmoid_weight_multiplication = execution_values[neuron_execution_values_start + 4] = linear_sigmoid * neuron_weights[neuron_weights_start + 1];
	data_t linear_tanh = device_tanh_activation(
		0,0,0,0,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		0, 5, 1
	);
	data_t store_gate_tanh_weight_multiplication = execution_values[neuron_execution_values_start + 6] = linear_tanh * neuron_weights[neuron_weights_start + 2];
	state[neuron_state_start] = execution_values[neuron_execution_values_start + 7] = cell_state += store_gate_sigmoid_weight_multiplication * store_gate_tanh_weight_multiplication;
	
	// Output Gate
	data_t cell_state_tanh = device_tanh_activation(
		0, 0, 0, false,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		7, 8, true
	);
	data_t output_gate_sigmoid_weight_multiplication = execution_values[neuron_execution_values_start + 9] = linear_sigmoid * neuron_weights[neuron_weights_start + 3];
	
	data_t output = output_gate_sigmoid_weight_multiplication * cell_state_tanh;
	state[neuron_state_start + 1] = activations[activations_start + layer_activations_start + tid] = output;
}

#endif
