
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
	size_t activations_start = t * neuron_count + (t + 1) * neuron_count * gaps_between_usable_arrays_t_count;
	size_t activation_read = activations_start + previous_layer_activations_start + tid % previous_layer_length;

	out_arr[tid] = activations[activation_read];
}

__global__ void get_pondered_activations_neat(
	size_t t_count, data_t *activations, size_t neuron_count, size_t layer_neuron_count, size_t connection_count,
	data_t *out_arr, size_t *connection_points, size_t *connection_neuron_i, data_t *weights, 
	size_t max_connection_count_at_layer, size_t gaps_between_usable_arrays_t_count
)
{
	size_t tid = get_tid();
	
	size_t n_outputs_per_t = max_connection_count_at_layer * layer_neuron_count;
	size_t out_len = n_outputs_per_t * t_count;
	if (tid >= out_len) return;
	out_arr[tid] = 0;
	if (tid >= connection_count * t_count) return;

	size_t t = tid / n_outputs_per_t;
	size_t connection_i = tid % connection_count;

	size_t activations_start = neuron_count * t + gaps_between_usable_arrays_t_count * neuron_count * (t + 1);
	size_t activations_i = activations_start + connection_points[connection_i];

	size_t write_i = 
		max_connection_count_at_layer * layer_neuron_count * t // Go to time step activations start
		+ max_connection_count_at_layer * connection_neuron_i[connection_i] // Go to layer activations
		+ connection_i;

	out_arr[write_i] = activations[activations_i] * weights[connection_i];
}

__global__ void insert_execution_values(
	size_t t_count, size_t nn_execution_value_count, size_t layer_neuron_count,
	size_t layer_execution_values_start, size_t execution_values_per_neuron, size_t neuron_execution_values_i,
	size_t gaps_between_usable_arrays_t_count,
	data_t *to_insert, data_t *execution_values)
{
	size_t tid = get_tid();
	size_t neurons_to_write = layer_neuron_count * t_count;
	if (tid >= neurons_to_write) return;

	size_t layer_neuron_i = tid % layer_neuron_count;

	size_t t = tid / layer_neuron_count;
	size_t execution_values_start = t * nn_execution_value_count + (t + 1) * nn_execution_value_count * gaps_between_usable_arrays_t_count;
	size_t write_i = execution_values_start + layer_execution_values_start + layer_neuron_i * execution_values_per_neuron + neuron_execution_values_i;

	execution_values[write_i] = to_insert[tid];
}
