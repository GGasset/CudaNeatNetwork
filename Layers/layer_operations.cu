
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

__global__ data_t LSTM_derivatives(
	size_t t_count, data_t *activations, data_t *execution_vals, data_t *derivatives, data_t *weights,
	nn_lens lens, layer_properties layer, size_t timestep_gap
)
{
	size_t total_neuron_count = t_count * layer.neuron_count;

	size_t tid = get_tid();
	if (tid >= total_neuron_count) return;

	size_t neuron_i = total_neuron_count % layer.neuron_count;
	size_t t = tid / layer.neuron_count;
	size_t arr_t = (t + 1) * timestep_gap + t;

	size_t neuron_weights_start = neuron_i * 4;
	size_t neuron_derivatives_start = arr_t * lens.derivative
		+ layer.derivatives_start + layer.derivatives_per_neuron * neuron_i;
	size_t execution_values_start = arr_t * lens.execution_values
		+ layer.execution_values_start + layer.execution_values_per_neuron * neuron_i;

	data_t forget_weight = weights[neuron_weights_start];
	data_t input_weight = weights[neuron_weights_start + 1];
	data_t candidate_cell_weight = weights[neuron_weights_start + 2];
	data_t output_weight = weights[neuron_weights_start + 3];

	data_t initial_cell_state = execution_vals[execution_values_start + 2];
	data_t output_cell_state = execution_vals[execution_values_start + 11];

	data_t initial_hidden_state = execution_vals[execution_values_start + 1];

	data_t previous_hidden_derivative_to_tanh = 1;
	data_t previous_hidden_derivative_to_sigmoid = 1;
	data_t previous_hidden_derivative_to_weight = 1;
	if (timestep_gap != 0)
	{
		size_t prev_derivatives_start = neuron_derivatives_start - lens.derivative;
		
		previous_hidden_derivative_to_tanh = derivatives[prev_derivatives_start + 21];
		previous_hidden_derivative_to_sigmoid = derivatives[prev_derivatives_start + 22];
		previous_hidden_derivative_to_weight = derivatives[prev_derivatives_start + 23];
	}
	derivatives[neuron_derivatives_start] = previous_hidden_derivative_to_tanh;
	derivatives[neuron_derivatives_start + 1] = previous_hidden_derivative_to_sigmoid;
	derivatives[neuron_derivatives_start + 2] = previous_hidden_derivative_to_weight;

	data_t linear_hidden = execution_vals[execution_values_start];
	data_t sigmoid_lh = execution_vals[execution_values_start + 3];
	data_t tanh_lh = execution_vals[execution_values_start + 6];

	data_t sigmoid_lh_derivative = sigmoid_derivative(linear_hidden);
	data_t tanh_lh_derivative = tanh_derivative(linear_hidden);

	derivatives[neuron_derivatives_start + 3] = sigmoid_lh_derivative;
	derivatives[neuron_derivatives_start + 4] = tanh_lh_derivative;

	// Forget Gate
	data_t forget_weight_partial_derivative = sigmoid_lh;
	data_t forget_sigmoid_partial_derivative = sigmoid_lh_derivative * forget_weight;

	derivatives[neuron_derivatives_start + 5] = forget_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 6] = forget_sigmoid_partial_derivative;


	data_t forget_output = execution_vals[execution_values_start + 4];
	data_t forget_out_partial_derivative_to_weight = forget_weight_partial_derivative * initial_cell_state;
	data_t forget_out_partial_derivative_to_sigmoid = forget_sigmoid_partial_derivative * initial_cell_state;
	//data_t initial_cell_partial_derivative = previous_cell_derivative * forget_output;

	derivatives[neuron_derivatives_start + 7] = forget_out_partial_derivative_to_weight;
	derivatives[neuron_derivatives_start + 8] = forget_out_partial_derivative_to_sigmoid;
	derivatives[neuron_derivatives_start + 9] = forget_output; 
			// Forget output to calculate initial cell gradient depending on path taken. To calculate multiply by the previous t output cell derivative, addition to inital cell or to store gate, else 0

	// Store Gate
	data_t input_weight_output = execution_vals[execution_values_start + 5];
	data_t candidate_weight_output = execution_vals[execution_values_start + 7];

	// Input Gate
	data_t input_weight_partial_derivative = sigmoid_lh;
	data_t input_sigmoid_partial_derivative = sigmoid_lh_derivative * input_weight;

	derivatives[neuron_derivatives_start + 10] = input_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 11] = input_sigmoid_partial_derivative;

	// Candidate Cell Gate
	data_t candidate_weight_partial_derivative = tanh_lh;
	data_t candidate_tanh_partial_derivative = tanh_lh_derivative * candidate_cell_weight;

	derivatives[neuron_derivatives_start + 12] = candidate_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 13] = candidate_tanh_partial_derivative;

	//  Store output
	data_t store_mult_input_gate_partial_derivative_to_sigmoid = input_sigmoid_partial_derivative * candidate_weight_output;
	data_t store_mult_input_gate_partial_derivative_to_weight = input_weight_partial_derivative * candidate_weight_output;
	data_t store_mult_candidate_gate_partial_derivative_to_tanh = candidate_tanh_partial_derivative * input_weight_output;
	data_t store_mult_candidate_gate_partial_derivative_to_weight = candidate_weight_partial_derivative * input_weight_output;

	derivatives[neuron_derivatives_start + 14] = store_mult_input_gate_partial_derivative_to_sigmoid;
	derivatives[neuron_derivatives_start + 15] = store_mult_input_gate_partial_derivative_to_weight;
	derivatives[neuron_derivatives_start + 16] = store_mult_candidate_gate_partial_derivative_to_tanh;
	derivatives[neuron_derivatives_start + 17] = store_mult_candidate_gate_partial_derivative_to_weight;

	// Output Gate
	data_t output_tanh = execution_vals[execution_values_start + 8];
	data_t output_weight_multiplication = execution_vals[execution_values_start + 9];

	data_t output_tanh_derivative = tanh_derivative(output_cell_state);
	derivatives[neuron_derivatives_start + 18] = output_tanh_derivative;

	data_t output_weight_partial_derivative = sigmoid_lh;
	data_t output_weight_sigmoid_partial_derivative = sigmoid_lh_derivative * output_weight;

	derivatives[neuron_derivatives_start + 19] = output_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 20] = output_weight_sigmoid_partial_derivative;

	data_t output_multiplication_partial_derivative_to_weight = output_tanh * output_weight_partial_derivative;
	data_t output_multiplication_partial_derivative_to_sigmoid = output_tanh * output_weight_sigmoid_partial_derivative;
	data_t output_multiplication_partial_derivative_to_tanh = output_weight_multiplication * output_tanh_derivative;

	derivatives[neuron_derivatives_start + 21] = output_multiplication_partial_derivative_to_tanh;
	derivatives[neuron_derivatives_start + 22] = output_multiplication_partial_derivative_to_sigmoid;
	derivatives[neuron_derivatives_start + 23] = output_multiplication_partial_derivative_to_weight;
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

__global__ void backpropagate_LSTM(
	size_t execution_lines, data_t *gradients, data_t *costs, data_t *derivatives,
	layer_properties layer, nn_lens lens, size_t t_count, size_t execution_line_t
)
{
	size_t total_neuron_count = execution_lines * layer.neuron_count;

	size_t tid = get_tid();
	if (tid >= total_neuron_count) return ;

	size_t neuron_i = tid % layer.neuron_count;

	size_t t = tid / layer.neuron_count; 

	//	           initial gaps    +other_execs+gap until next execution line
	size_t arr_t = execution_line_t * (t + 1) +     t     + t * (t_count - execution_line_t - 1);

	size_t neuron_derivatives_start = arr_t * lens.derivative
		+ layer.derivatives_start + layer.derivatives_per_neuron * neuron_i;

	size_t connections_gradients_start = arr_t * lens.gradients
		+ layer.gradients_start
		+ layer.per_neuron_gradients_start[neuron_i];
	size_t neuron_gradients_start =
		connections_gradients_start + layer.per_connection_gradient_count[neuron_i];
	
	data_t next_hidden_state_gradient = 0;
	data_t next_cell_state_gradient = 0;
	if (execution_line_t < t_count - 1)
	{
		size_t next_gradient_start = neuron_gradients_start + lens.gradients;

		next_hidden_state_gradient = gradients[next_gradient_start + 5];
		next_cell_state_gradient  = gradients[next_gradient_start + 4];
	}

	// Output Losses
	data_t cost = costs[arr_t * lens.neurons + layer.activations_start + neuron_i];
	data_t output_hidden_gradient_to_tanh = (cost 
									- next_hidden_state_gradient * derivatives[neuron_derivatives_start + 21]);
																// output_multiplication_partial_derivative_to_tanh

	data_t output_hidden_gradient_to_sigmoid = (cost
									- next_hidden_state_gradient * derivatives[neuron_derivatives_start + 22]);
																// output_multiplication_partial_derivative_to_sigmoid

	data_t output_hidden_gradient_to_weight = (cost
									- next_hidden_state_gradient * derivatives[neuron_derivatives_start + 23]);
																// output_multiplication_partial_derivative_to_weight

	// To cell state
	data_t output_cell_gradient_to_cell_state = output_hidden_gradient_to_tanh * derivatives[neuron_derivatives_start + 21];
												// output_multiplication_partial_derivative_to_tanh
												// multiplied twice due to previous t linear hidden sum partial derivative

	output_cell_gradient_to_cell_state *= derivatives[neuron_derivatives_start + 18]; // cell tanh derivative
	output_cell_gradient_to_cell_state -= next_cell_state_gradient;

	// To previous cell state
	data_t previous_cell_state_gradient = output_cell_gradient_to_cell_state;
								//  forget_weight_multiplication output
								//  store addition partial derivative
	previous_cell_state_gradient *= derivatives[neuron_derivatives_start + 9];

								//  forget multiplication partial derivative
	previous_cell_state_gradient *= derivatives[neuron_derivatives_start + 9];
	gradients[neuron_gradients_start + 4] = previous_cell_state_gradient;


	// output weight gradient
	data_t output_weight_gradient = output_hidden_gradient_to_weight;
	output_weight_gradient *= derivatives[neuron_derivatives_start + 23];// output multiplication to weight
	output_weight_gradient *= derivatives[neuron_derivatives_start + 19];// weight partial derivative
	gradients[neuron_gradients_start + 3] = output_weight_gradient;


	// Output gate to linear hidden
	data_t output_gate_sigmoid_gradient = output_hidden_gradient_to_sigmoid;
								//  output gate multiplication to sigmoid partial derivative
	output_gate_sigmoid_gradient *= derivatives[neuron_derivatives_start + 22];
	output_gate_sigmoid_gradient *= derivatives[neuron_derivatives_start + 20];// weight to sigmoid partial derivative
	output_gate_sigmoid_gradient *= derivatives[neuron_derivatives_start + 3]; // sigmoid_lh derivative


	// Store Gate
	//	To candidate_weight
	data_t candidate_weight_gradient = output_cell_gradient_to_cell_state;
	candidate_weight_gradient *= derivatives[neuron_derivatives_start + 17];
	candidate_weight_gradient *= derivatives[neuron_derivatives_start + 17];
	candidate_weight_gradient *= derivatives[neuron_derivatives_start + 12];
	gradients[neuron_gradients_start + 2] = candidate_weight_gradient;

	//	To input weight
	data_t input_weight_gradient = output_cell_gradient_to_cell_state;
	input_weight_gradient *= derivatives[neuron_derivatives_start + 15];
	input_weight_gradient *= derivatives[neuron_derivatives_start + 15];
	input_weight_gradient *= derivatives[neuron_derivatives_start + 10];
	gradients[neuron_gradients_start + 1] = input_weight_gradient;

	//	To linear hidden
	data_t store_gate_sigmoid_gradient = output_cell_gradient_to_cell_state;
	store_gate_sigmoid_gradient *= derivatives[neuron_derivatives_start + 14];
	store_gate_sigmoid_gradient *= derivatives[neuron_derivatives_start + 14];
	store_gate_sigmoid_gradient *= derivatives[neuron_derivatives_start + 11];
	store_gate_sigmoid_gradient *= derivatives[neuron_derivatives_start + 3];

	data_t store_gate_tanh_gradient = output_cell_gradient_to_cell_state;
	store_gate_tanh_gradient *= derivatives[neuron_derivatives_start + 16];
	store_gate_tanh_gradient *= derivatives[neuron_derivatives_start + 16];
	store_gate_tanh_gradient *= derivatives[neuron_derivatives_start + 13];
	store_gate_tanh_gradient *= derivatives[neuron_derivatives_start + 4];


	// Forget Gate
	//	Weight
	data_t forget_weight_gradient = output_cell_gradient_to_cell_state;
	forget_weight_gradient *= derivatives[neuron_derivatives_start + 7];
	forget_weight_gradient *= derivatives[neuron_derivatives_start + 7];
	forget_weight_gradient *= derivatives[neuron_derivatives_start + 5];
	gradients[neuron_gradients_start] = forget_weight_gradient;

	//	To linear hidden
	data_t forget_sigmoid_gradient = output_cell_gradient_to_cell_state;
	forget_sigmoid_gradient *= derivatives[neuron_derivatives_start + 8];
	forget_sigmoid_gradient *= derivatives[neuron_derivatives_start + 8];
	forget_sigmoid_gradient *= derivatives[neuron_derivatives_start + 6];


	// Linear hidden
	data_t linear_hidden_gradient = -(forget_sigmoid_gradient + store_gate_sigmoid_gradient + store_gate_tanh_gradient + output_gate_sigmoid_gradient);
	gradients[neuron_gradients_start + 5] = linear_hidden_gradient;

	data_t linear_function_gradient = linear_hidden_gradient * derivatives[neuron_derivatives_start];
															// linear function derivative
	gradients[connections_gradients_start] = linear_function_gradient;
}
