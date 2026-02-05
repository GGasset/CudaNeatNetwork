
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
	}

	g_activation_function n_threads(t_count * layer.neuron_count) (
		t_count, execution_vals, activations, activation, layer, lens, timestep_gap
	);
	cudaDeviceSynchronize();
}
