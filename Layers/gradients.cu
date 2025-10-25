#include "gradients.cuh"
__global__ void LSTM_gradient_calculation(
	data_t* derivatives, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* gradients, size_t gradients_start, size_t next_t_gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	data_t* costs, size_t costs_start, size_t layer_costs_start,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t neuron_derivatives_start = derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;

	size_t connections_gradients_start = gradients_start + layer_gradients_start + neuron_gradients_starts[tid];
	size_t neuron_gradients_start = connections_gradients_start + connection_associated_gradient_counts[tid];

	size_t next_connections_gradients_start = next_t_gradients_start + layer_gradients_start + neuron_gradients_starts[tid];
	size_t next_neuron_gradients_start = next_connections_gradients_start + connection_associated_gradient_counts[tid];


	data_t next_hidden_state_gradient = 0;
	data_t next_cell_state_gradient = 0;
	if (next_t_gradients_start)
	{
		next_hidden_state_gradient = gradients[next_neuron_gradients_start + 5];
		next_cell_state_gradient  = gradients[next_neuron_gradients_start + 4];
	}

	// Output Losses
	data_t output_gradient = costs[costs_start + layer_costs_start + tid];
	data_t output_hidden_gradient_to_tanh = (output_gradient 
									- next_hidden_state_gradient * derivatives[neuron_derivatives_start + 21]);
																// output_multiplication_partial_derivative_to_tanh

	data_t output_hidden_gradient_to_sigmoid = (output_gradient
									- next_hidden_state_gradient * derivatives[neuron_derivatives_start + 22]);
																// output_multiplication_partial_derivative_to_sigmoid

	data_t output_hidden_gradient_to_weight = (output_gradient
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

__global__ void LSTM_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	field_t* neuron_weights,
	gradient_hyperparameters hyperparameters, Optimizers optimizer,
	size_t layer_length, size_t connections_weight_count
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t layer_neuron_gradient_start_i = neuron_gradients_starts[tid] + connection_associated_gradient_counts[tid];
	size_t neuron_gradients_start_i = gradients_start + layer_gradients_start + layer_neuron_gradient_start_i;
	size_t neuron_weights_start = static_cast<size_t>(4) * tid;

	for (size_t i = 0; i < 4; i++)
		optimizer.subtract_gradient(
			neuron_weights + neuron_weights_start + i, connections_weight_count + neuron_weights_start + i, 
			gradients[neuron_gradients_start_i + i], 
			hyperparameters
		);
}

__global__ void global_neuron_gradient_calculation(
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start, size_t layer_costs_start,
	ActivationFunctions activation,
	size_t layer_length,
	data_t *vars
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	data_t input_gradient = costs[costs_start + layer_costs_start + tid];
	data_t activation_input = execution_values[execution_values_start + execution_values_layer_start + tid];
	data_t bias_gradient = input_gradient;
	switch (activation)
	{
	case sigmoid:
		bias_gradient *= device_sigmoid_derivative(activation_input);
		break;
	case _tanh:
		bias_gradient *= device_tanh_derivative(activation_input);
		break;
	default:
		break;
	}
	size_t gradient_write_i = gradients_start + layer_gradients_start + neuron_gradients_starts[tid];
	gradients[gradient_write_i] = bias_gradient;
}

__host__ void neuron_gradient_calculation(
	data_t *execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	data_t *gradients, size_t gradients_start, size_t layer_gradients_start, size_t *neuron_gradients_starts,
	data_t *costs, size_t costs_start, size_t layer_costs_start, 
	ActivationFunctions activation, 
	size_t layer_length)
{
	switch (activation)
	{
	case softmax:
	{
		data_t *linear_funcs = host_extract_execution_values(
			execution_values + execution_values_start + execution_values_layer_start, layer_length,
			execution_values_per_neuron, 0
		);
		apply_func<data_t, float, float> n_threads(layer_length) (linear_funcs, layer_length, exp);
		cudaDeviceSynchronize();
		data_t exponent_sum = cuda_sum(linear_funcs, layer_length);
		cudaFree(linear_funcs);

		global_neuron_gradient_calculation n_threads(layer_length) (
			execution_values, execution_values_start, execution_values_layer_start,
			gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
			costs, costs_start, layer_costs_start,
			activation,
			layer_length,
			&exponent_sum
		);
		break;
	}
	
	default:
		global_neuron_gradient_calculation n_threads(layer_length) (
			execution_values, execution_values_start, execution_values_layer_start,
			gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
			costs, costs_start, layer_costs_start,
			activation,
			layer_length,
			0
		);
		break;
	}
}

__global__ void cud_set_dropout(
	float dropout_rate, float* normalized_random_samples, short* dropout,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t i = tid;
	dropout[i] = normalized_random_samples[i] > dropout_rate;
}
