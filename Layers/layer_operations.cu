
#include "layer_operations.cuh"

__device__ data_t sigmoid_activation(data_t in)
{
	return 1 / (1 + exp(-in));
}

__device__ data_t tanh_activation(data_t in)
{
	in = device_clip(in, -10, 10);
	data_t exp_x = exp(in);
	data_t exp_minus_x = exp(-in);
	return (exp_x - exp_minus_x) / (exp_x + exp_minus_x);
}

__device__ data_t sofmax_activation(data_t in, data_t exponent_sum)
{
	exponent_sum += 1e-7;
	return exp(in) / exponent_sum;
}

__global__ void g_activation_function(
	size_t t_count, data_t *execution_vals, data_t *activations,
	ActivationFunctions activation, layer_properties layer, nn_lens lens, size_t timestep_gap
)
{
	size_t tid = get_tid();
	size_t total_neuron_count = t_count * layer.neuron_count;

	if (tid >= total_neuron_count) return;

	size_t t = tid / layer.neuron_count;
	size_t neuron_i = tid % layer.neuron_count;

	size_t arrays_t = (t * 2 + 1);
	size_t execution_values_start = arrays_t * lens.execution_values;
	size_t neuron_execution_values_start = execution_values_start + layer.execution_values_start + layer.execution_values_per_neuron * neuron_i;

	data_t in = execution_vals[neuron_execution_values_start];
	data_t out = 0;

	switch (activation)
	{
	case sigmoid:
		out = sigmoid_activation(in);
		break;
	case _tanh:
		out = tanh_activation(in);
		break;
	case softmax:
		{
			data_t exponent_sum = execution_vals[neuron_execution_values_start + 1];
			out = sofmax_activation(in, exponent_sum);
		}
		break;
	case no_activation:
		out = in;
		break;
	
	default: return;
	}

	size_t neuron_activation_i = arrays_t * lens.neurons + layer.activations_start + neuron_i;
	activations[neuron_activation_i] = out;
}

__host__ void activation_function(
	size_t t_count, data_t *execution_vals, data_t *activations,
	ActivationFunctions activation, layer_properties layer, nn_lens lens, size_t timestep_gap)
{
	if (activation == softmax)
	{
		size_t total_neuron_n = t_count * layer.neuron_count;
		data_t *linear_functions = 0;
		cudaMalloc(&linear_functions, sizeof(data_t) * total_neuron_n);
		network_value_extract n_threads(total_neuron_n) (
			t_count, lens.execution_values, layer.neuron_count, layer.execution_values_per_neuron,
			timestep_gap, layer.execution_values_start, 0, 1, execution_vals, linear_functions
		);
		cudaDeviceSynchronize();
		exp_arr n_threads(total_neuron_n) (linear_functions, total_neuron_n);
		cudaDeviceSynchronize();

		data_t *exponent_sums = multi_PRAM_add(linear_functions, layer.neuron_count, t_count);
		data_t *expanded_sums = linear_functions;
		clone_arr_values_n_times n_threads(total_neuron_n) (
			exponent_sums, t_count, layer.neuron_count, expanded_sums, total_neuron_n
		);
		cudaDeviceSynchronize();
		cudaFree(exponent_sums);

		network_value_insert n_threads(total_neuron_n) (
			t_count, lens.execution_values, layer.neuron_count, layer.execution_values_per_neuron, timestep_gap,
			layer.execution_values_start, 1, 1, expanded_sums, execution_vals
		);
		cudaDeviceSynchronize();
		cudaFree(linear_functions);
	}

	g_activation_function n_threads(t_count * layer.neuron_count) (
		t_count, execution_vals, activations, activation, layer, lens, timestep_gap
	);
	cudaDeviceSynchronize();
}

__global__ void LSTM_execution(
	size_t t_count, data_t *execution_vals, data_t *activations, data_t *weights,
	layer_properties layer, nn_lens lengths, size_t timestep_gap
)
{
	size_t tid = get_tid();

	size_t total_neuron_count = layer.neuron_count * t_count;
	if (tid >= total_neuron_count) return;

	size_t neuron_i = tid % layer.neuron_count;
	size_t weights_start = neuron_i * 4;

	// Parallel execution line i
	size_t t = tid / layer.neuron_count;
	size_t array_t = (t + 1) * timestep_gap + t;

	size_t execution_val_i = 
		array_t * lengths.execution_values
		+ layer.execution_values_start
		+ layer.execution_values_per_neuron * neuron_i;

	data_t hidden_state = execution_vals[execution_val_i + 1];
	data_t cell_state = execution_vals[execution_val_i + 2];
	if (cell_state == 0 && hidden_state == 0 && timestep_gap != 0)
	{
		hidden_state = execution_vals[execution_val_i - lengths.execution_values + 10];
		execution_vals[execution_val_i + 1] = hidden_state;

		cell_state = execution_vals[execution_val_i - lengths.execution_values + 11];
		execution_vals[execution_val_i + 2] = cell_state;
	}
	data_t linear_hidden = hidden_state + execution_vals[execution_val_i];
	execution_vals[execution_val_i] = linear_hidden;

	data_t linear_sigmoid = sigmoid_activation(linear_hidden);
	execution_vals[execution_val_i + 3] = linear_sigmoid;

	// Forget_gate
	data_t forget_gate_out = linear_sigmoid * weights[weights_start];
	execution_vals[execution_val_i + 4] = forget_gate_out;
	cell_state *= forget_gate_out;

	// Store Gate
	data_t store_sigmoid_weight_out = linear_sigmoid * weights[weights_start + 1];
	execution_vals[execution_val_i + 5] = store_sigmoid_weight_out;

	data_t linear_tanh = tanh_activation(linear_hidden);
	execution_vals[execution_val_i + 6] = linear_tanh;

	data_t store_tanh_weight_out = linear_tanh * weights[weights_start + 2];
	execution_vals[execution_val_i + 7] = store_tanh_weight_out;

	cell_state += store_sigmoid_weight_out * store_tanh_weight_out;

	// Out Gate
	data_t cell_state_tanh = tanh_activation(cell_state);
	execution_vals[execution_val_i + 8] = cell_state_tanh;

	data_t out_weight_out = linear_sigmoid * weights[weights_start + 3];
	execution_vals[execution_val_i + 9] = out_weight_out;

	hidden_state = out_weight_out * cell_state_tanh;
	execution_vals[execution_val_i + 10] = hidden_state;
	execution_vals[execution_val_i + 11] = cell_state;

	size_t activations_start_i = array_t * lengths.neurons
		+ layer.activations_start + neuron_i;
	activations[activations_start_i] = hidden_state;
}

__device__ data_t sigmoid_derivative(data_t in)
{
	data_t exp_minus_x = exp(-in);
	return (exp_minus_x) / ((1 +exp_minus_x) * (1 + exp_minus_x));
}

__device__ data_t tanh_derivative(data_t in)
{
	in = device_clip(in, -10, 10);
	data_t exp_2_x = exp(in * 2);
	return (4 * exp_2_x) / ((exp_2_x + 1) * (exp_2_x + 1));
}

__device__ data_t sofmax_derivative(data_t in, data_t exponent_sum)
{
	return (exp(in) * (exponent_sum - exp(in))) / (exponent_sum * exponent_sum);
}

__global__ void backpropagate_activation(
	size_t t_count, data_t *execution_vals, data_t *gradients, data_t *costs,
	ActivationFunctions activation, layer_properties layer, nn_lens lens, size_t timestep_gap
)
{
	size_t tid = get_tid();

	size_t total_neuron_n = t_count * layer.neuron_count;
	if (tid >= total_neuron_n) return;

	size_t neuron_i = tid % layer.neuron_count;
	size_t t = tid / layer.neuron_count;
	size_t array_t = t + (t + 1) * timestep_gap;

	size_t neuron_execution_values_start = lens.execution_values * array_t + layer.execution_values_start 
										 + layer.execution_values_per_neuron * neuron_i;

	data_t linear_func = execution_vals[neuron_execution_values_start];

	size_t neuron_cost_i = lens.neurons * array_t + layer.activations_start + neuron_i;
	data_t cost = costs[neuron_cost_i];

	data_t bias_gradient = cost;
	switch (activation)
	{
	case sigmoid:
		bias_gradient *= sigmoid_derivative(linear_func);
		break;
	case _tanh:
		bias_gradient *= tanh_derivative(linear_func);
		break;
	case softmax:
		bias_gradient *= sofmax_derivative(linear_func, execution_vals[neuron_execution_values_start + 1]);
		break;
		
	case no_activation:
	default: break;
	}

	size_t neuron_gradients_start = lens.gradients * array_t + layer.gradients_start + layer.per_neuron_gradients_start[neuron_i];
	gradients[neuron_gradients_start] = bias_gradient;
}
