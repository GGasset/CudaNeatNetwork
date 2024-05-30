#ifndef DENSE_CONNECTIONS
#define DENSE_CONNETIONS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DenseConnections.h"

DenseConnections::DenseConnections(size_t previous_layer_activations_start, size_t previous_layer_length)
{
	this->previous_layer_activations_start = previous_layer_activations_start;
	this->previous_layer_length = previous_layer_length;
}

void DenseConnections::linear_function(size_t activations_start, data_t* activations,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
)
{
	cud_dense_linear_function kernel(dim3(previous_layer_length / 32 + (previous_layer_length % 32 > 0), neuron_count, 1), 32) (
		previous_layer_length, weights,
		activations_start, previous_layer_activations_start, activations,
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cud_add_biases kernel(dim3(neuron_count / 32 + (neuron_count % 32 > 0), 1, 1), 32) (
		neuron_count, biases,
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cudaDeviceSynchronize();
}

void DenseConnections::calculate_derivative(
	size_t activations_start, data_t* activations,
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
)
{
	cud_dense_linear_function_derivative kernel(dim3(previous_layer_length / 32 + (previous_layer_length % 32 > 0), neuron_count, 1), 32) (
		activations_start, previous_layer_activations_start, previous_layer_length, activations,
		derivatives_start, derivatives_layer_start, derivatives_per_neuron, derivatives,
		weights
	);
	cud_add_bias_derivative kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		neuron_count, derivatives_start, derivatives_layer_start, derivatives_per_neuron, derivatives
	);
}

void DenseConnections::calculate_gradients(
	data_t* activations, size_t activations_start, 
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, 
	data_t* costs, size_t costs_start
)
{
	cud_dense_gradient_calculation kernel(dim3(previous_layer_length / 32 + (previous_layer_length % 32 > 0), neuron_count), 32) (
		activations, activations_start,
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start,
		previous_layer_activations_start, previous_layer_length, weights
	);
}

void DenseConnections::subtract_gradients(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t learning_rate, short* dropout, data_t gradient_clip
)
{
	cud_dense_gradient_subtraction kernel(dim3(previous_layer_length / 32 + (previous_layer_length % 32 > 0), neuron_count), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		weights, previous_layer_length, learning_rate, dropout, gradient_clip
	);
	bias_gradient_subtraction kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		biases, neuron_count, learning_rate, dropout, gradient_clip
	);
}
#endif