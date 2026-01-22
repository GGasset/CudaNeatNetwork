
#include "connections_functionality.cuh"

__global__ void extract_activations_dense(
	size_t t_count, data_t *activations, size_t neuron_count, size_t layer_neuron_count, 
	data_t *out_arr, size_t previous_layer_activations_start, size_t previous_layer_length, 
	size_t gaps_between_usable_arrays_t_count
)
{
	size_t tid = get_tid();
	size_t per_t_input_count = previous_layer_length * layer_neuron_count;
	size_t input_count = per_t_input_count * t_count;

	if (tid >= input_count) return;

	size_t t = tid % t_count;
	size_t activations_start = t * neuron_count + t * neuron_count * gaps_between_usable_arrays_t_count;
	size_t activation_read = activations_start + previous_layer_activations_start + tid % previous_layer_length;

	out_arr[tid] = activations[activation_read];
}

__global__ void insert_execution_values(
	size_t t_count, size_t nn_execution_value_count, size_t layer_neuron_count, 
	size_t layer_execution_values_start, size_t execution_values_per_neuron, size_t neuron_execution_values_i, 
	size_t gaps_between_usable_arrays_t_count, 
	data_t *to_insert, data_t *execution_values
)
{
	size_t tid = get_tid();
	size_t neurons_to_write = layer_neuron_count * t_count;
	if (tid >= neurons_to_write) return;

	size_t layer_neuron_i = tid % layer_neuron_count;

	size_t execution_values_start = t_count * nn_execution_value_count + t_count * nn_execution_value_count * gaps_between_usable_arrays_t_count;
	size_t write_i = execution_values_start + layer_execution_values_start + layer_neuron_i * execution_values_per_neuron + neuron_execution_values_i;

	execution_values[write_i] = to_insert[tid];
}
