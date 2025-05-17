#include "connection_gradients.cuh"

__global__ void cud_dense_gradient_calculation(
	data_t* activations, size_t activations_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start,
	size_t previous_layer_activations_start, size_t previous_layer_length,
	field_t* weights
)
{
	size_t tid = get_tid();
	if (tid >= previous_layer_length)
		return;

	// Input gradient is bias gradient
	size_t input_gradient_i = gradients_start + layer_gradients_start + neuron_gradients_starts[blockIdx.y];
	data_t input_gradient = gradients[input_gradient_i];
	size_t weight_gradient_i = input_gradient_i + tid + 1;
	field_t weight = weights[tid];
	data_t activation = activations[activations_start + previous_layer_activations_start + tid];
	gradients[weight_gradient_i] = input_gradient * activation;
	atomicAdd(costs + costs_start + previous_layer_activations_start + tid, -input_gradient * weight);
}

__global__ void cud_NEAT_gradient_calculation(
	data_t* activations, size_t activations_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start,
	size_t connection_count, field_t* weights, size_t* connection_points, size_t* connection_neuron_i
)
{
	size_t tid = get_tid();
	if (tid >= connection_count)
		return;

	size_t neuron_i = connection_neuron_i[tid];
	size_t input_gradient_i = gradients_start + layer_gradients_start + neuron_gradients_starts[neuron_i];
	size_t weight_gradient_i = input_gradient_i + tid + 1;
	size_t connection_input_i = connection_points[tid];

	data_t input_gradient = gradients[input_gradient_i];
	gradients[weight_gradient_i] = input_gradient * activations[activations_start + connection_input_i];
	atomicAdd(costs + costs_start + connection_input_i, -input_gradient * weights[tid]);
}

__global__ void bias_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	field_t* biases, size_t layer_length, gradient_hyperparameters hyperparameter, IOptimizer* optimizer
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t gradient_i = gradients_start + layer_gradients_start + neuron_gradients_starts[tid];
	data_t gradient = gradients[gradient_i];
	optimizer->hyperparameter_subtract_gradient(biases + tid, gradient, neuron_gradients_starts[tid], hyperparameter);
}

__global__ void cud_dense_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	field_t* weights, size_t previous_layer_length,
	gradient_hyperparameters hyperparameter, IOptimizer* optimizer
)
{
	size_t tid = get_tid();
	if (tid >= previous_layer_length) return;

	size_t layer_gradient_i = neuron_gradients_starts[blockIdx.y] + tid + 1;
	size_t gradient_i = gradients_start + layer_gradients_start + layer_gradient_i;
	data_t gradient = gradients[gradient_i];
	size_t weight_i = previous_layer_length * blockIdx.y + tid;
	optimizer->hyperparameter_subtract_gradient(weights + weight_i, gradient, layer_gradient_i, hyperparameter);
}

__global__ void cud_NEAT_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	size_t* connection_neuron_i, size_t connection_count, 
	field_t* weights,
	gradient_hyperparameters hyperparameter, IOptimizer* optimizer
)
{
	size_t tid = get_tid();
	if (tid >= connection_count) return;

	size_t neuron_i = connection_neuron_i[tid];

	size_t layer_gradient_i = neuron_gradients_starts[neuron_i] + tid + 1;
	size_t gradient_i = gradients_start + layer_gradients_start + layer_gradient_i;
	data_t gradient = gradients[gradient_i];
	optimizer->hyperparameter_subtract_gradient(weights + tid, gradient, layer_gradient_i, hyperparameter);
}
