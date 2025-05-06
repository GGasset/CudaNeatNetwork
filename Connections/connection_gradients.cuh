#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"
#include "cuda_functionality.cuh"

__global__ void cud_dense_gradient_calculation(
	data_t* activations, size_t activations_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start,
	size_t previous_layer_activations_start, size_t previous_layer_length,
	field_t* weights
);

__global__ void cud_NEAT_gradient_calculation(
	data_t* activations, size_t activations_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start,
	size_t connection_count, field_t* weights, size_t* connection_points, size_t* connection_neuron_i
);

__global__ void bias_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	field_t* biases, size_t layer_length, data_t learning_rate, data_t max_subtracted_gradient
);

__global__ void cud_dense_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	field_t* weights, size_t previous_layer_length, 
	data_t learning_rate, data_t max_subtracted_gradient
);

__global__ void cud_NEAT_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	size_t* connection_neuron_i, size_t connection_count, 
	field_t* weights,
	data_t learning_rate, data_t max_subtracted_gradient
);
