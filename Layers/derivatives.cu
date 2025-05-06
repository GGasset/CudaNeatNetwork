#include "derivatives.cuh"

#include <cmath>

__device__ data_t device_sigmoid_derivative(
	data_t input
)
{
	data_t exp_minus_x = exp(-input);
	return (exp_minus_x) / ((1 +exp_minus_x) * (1 + exp_minus_x));
}

__device__ data_t device_tanh_derivative(
	data_t input
)
{
	data_t exp_2_x = exp(input * 2);
	return (4 * exp_2_x) / ((exp_2_x + 1) * (exp_2_x + 1));
}

__global__ void LSTM_derivative_calculation(
	data_t* prev_state_derivatives, data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t neuron_execution_values_start = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid;

	size_t neuron_weights_start = static_cast<size_t>(4) * tid;

	size_t neuron_derivatives_start = derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;
	size_t previous_neuron_derivatives_start = previous_derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;


	data_t initial_cell_state = execution_values[execution_values_start + 3];
	data_t output_cell_state = execution_values[execution_values_start + 7];

	data_t forget_weight = neuron_weights[neuron_weights_start];
	data_t input_weight = neuron_weights[neuron_weights_start + 1];
	data_t candidate_cell_weight = neuron_weights[neuron_weights_start + 2];
	data_t output_weight = neuron_weights[neuron_weights_start + 3];


	data_t linear_function_derivative = derivatives[neuron_derivatives_start];

	data_t previous_hidden_derivative_to_tanh = prev_state_derivatives[tid * 3];
	data_t previous_hidden_derivative_to_sigmoid = prev_state_derivatives[tid * 3 + 1];
	data_t previous_hidden_derivative_to_weight = prev_state_derivatives[tid * 3 + 2];
	if (derivatives_start != 0)
	{
		previous_hidden_derivative_to_tanh = derivatives[previous_neuron_derivatives_start + 21];
		previous_hidden_derivative_to_sigmoid = derivatives[previous_neuron_derivatives_start + 22];
		previous_hidden_derivative_to_weight = derivatives[previous_neuron_derivatives_start + 23];
	}
	derivatives[neuron_derivatives_start] = previous_hidden_derivative_to_tanh;
	derivatives[neuron_derivatives_start + 1] = previous_hidden_derivative_to_sigmoid;
	derivatives[neuron_derivatives_start + 2] = previous_hidden_derivative_to_weight;

	data_t linear_hidden = execution_values[neuron_execution_values_start];
	data_t sigmoid_lh = execution_values[neuron_execution_values_start + 1];
	data_t tanh_lh = execution_values[neuron_execution_values_start + 5];

	data_t sigmoid_lh_derivative = device_sigmoid_derivative(linear_hidden);
	data_t tanh_lh_derivative = device_tanh_derivative(linear_hidden);

	derivatives[neuron_derivatives_start + 3] = sigmoid_lh_derivative;
	derivatives[neuron_derivatives_start + 4] = tanh_lh_derivative;

	// Forget Gate
	data_t forget_weight_partial_derivative = sigmoid_lh;
	data_t forget_sigmoid_partial_derivative = sigmoid_lh_derivative * forget_weight;

	derivatives[neuron_derivatives_start + 5] = forget_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 6] = forget_sigmoid_partial_derivative;


	data_t forget_output = execution_values[neuron_execution_values_start + 2];
	data_t forget_out_partial_derivative_to_weight = forget_weight_partial_derivative * initial_cell_state;
	data_t forget_out_partial_derivative_to_sigmoid = forget_sigmoid_partial_derivative * initial_cell_state;
	//data_t initial_cell_partial_derivative = previous_cell_derivative * forget_output;

	derivatives[neuron_derivatives_start + 7] = forget_out_partial_derivative_to_weight;
	derivatives[neuron_derivatives_start + 8] = forget_out_partial_derivative_to_sigmoid;
	derivatives[neuron_derivatives_start + 9] = forget_output; // Forget output to calculate initial cell gradient depending on path taken. To calculate multiply by the previous t output cell derivative, addition to inital cell or to store gate, else 0


	// Store Gate
	data_t input_weight_output = execution_values[neuron_execution_values_start + 4];
	data_t candidate_weight_output = execution_values[neuron_execution_values_start + 6];

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
	data_t output_tanh = execution_values[neuron_execution_values_start + 8];
	data_t output_weight_multiplication = execution_values[neuron_execution_values_start + 9];

	data_t output_tanh_derivative = device_tanh_derivative(output_cell_state);
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
	prev_state_derivatives[tid * 3] = output_multiplication_partial_derivative_to_tanh;
	prev_state_derivatives[tid * 3 + 1] = output_multiplication_partial_derivative_to_sigmoid;
	prev_state_derivatives[tid * 3 + 2] = output_multiplication_partial_derivative_to_weight;
}

/*__global__ void LSTM_derivative_calculation(
	data_t* prev_state_derivatives, data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights, 
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t neuron_execution_values_start = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid;

	size_t neuron_weights_start = static_cast<size_t>(4) * tid;

	size_t neuron_derivatives_start = derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;
	size_t previous_neuron_derivatives_start = previous_derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;


	data_t initial_cell_state = execution_values[execution_values_start + 3];
	data_t output_cell_state = execution_values[execution_values_start + 7];

	data_t forget_weight = neuron_weights[neuron_weights_start];
	data_t input_weight = neuron_weights[neuron_weights_start + 1];
	data_t candidate_cell_weight = neuron_weights[neuron_weights_start + 2];
	data_t output_weight = neuron_weights[neuron_weights_start + 3];


	data_t linear_function_derivative = derivatives[neuron_derivatives_start];

	data_t previous_hidden_derivative = prev_state_derivatives[tid * 2];
	data_t previous_cell_derivative = prev_state_derivatives[tid * 2 + 1];
	if (derivatives_start != 0)
	{
		previous_hidden_derivative = derivatives[previous_neuron_derivatives_start + 3];
		previous_cell_derivative = derivatives[previous_neuron_derivatives_start + 4];
	}
	derivatives[neuron_derivatives_start + 1] = previous_hidden_derivative;
	derivatives[neuron_derivatives_start + 2] = previous_cell_derivative;

	data_t linear_hidden = execution_values[neuron_execution_values_start];
	data_t sigmoid_lh = execution_values[neuron_execution_values_start + 1];
	data_t tanh_lh = execution_values[neuron_execution_values_start + 5];

	data_t sigmoid_lh_derivative = device_sigmoid_derivative(linear_hidden);
	data_t tanh_lh_derivative = device_tanh_derivative(linear_hidden);

	derivatives[neuron_derivatives_start + 5] = sigmoid_lh_derivative;
	derivatives[neuron_derivatives_start + 6] = tanh_lh_derivative;



	// Forget gate
	data_t forget_weight_derivative = sigmoid_lh + sigmoid_lh_derivative * forget_weight;
	data_t forget_weight_partial_derivative = sigmoid_lh;
	data_t forget_sigmoid_partial_derivative = sigmoid_lh_derivative * forget_weight;

	derivatives[neuron_derivatives_start + 7] = forget_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 8] = forget_sigmoid_partial_derivative;


	data_t forget_output = execution_values[neuron_execution_values_start + 2];
	data_t forget_out_mult_derivative = previous_cell_derivative * forget_output;
	data_t forget_out_partial_derivative = forget_weight_derivative * initial_cell_state;
	data_t initial_cell_partial_derivative = previous_cell_derivative * forget_output;

	derivatives[neuron_derivatives_start + 9] = forget_out_partial_derivative;
	derivatives[neuron_derivatives_start + 10] = initial_cell_partial_derivative;



	// Store gate
	data_t input_weight_output = execution_values[neuron_execution_values_start + 4];
	data_t candidate_weight_output = execution_values[neuron_execution_values_start + 6];

	//  Input gate
	data_t input_weight_derivative = sigmoid_lh + sigmoid_lh_derivative * input_weight;
	data_t input_weight_partial_derivative = sigmoid_lh;
	data_t input_sigmoid_partial_derivative = sigmoid_lh_derivative * input_weight;

	derivatives[neuron_derivatives_start + 11] = input_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 12] = input_sigmoid_partial_derivative;

	//  Candidate cell gate
	data_t candidate_weight_derivative = tanh_lh + tanh_lh_derivative * candidate_cell_weight;
	data_t candidate_weight_partial_derivative = tanh_lh;
	data_t candidate_tanh_partial_derivative = tanh_lh_derivative * candidate_cell_weight;

	derivatives[neuron_derivatives_start + 13] = candidate_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 14] = candidate_tanh_partial_derivative;

	//  Store output
	data_t store_multiplication_derivative = input_weight_derivative * candidate_weight_output + candidate_weight_derivative * input_weight_output;
	data_t store_mult_input_gate_partial_derivative = input_weight_derivative * candidate_weight_output;
	data_t store_mult_candidate_gate_partial_derivative = candidate_weight_derivative * input_weight_output;

	derivatives[neuron_derivatives_start + 15] = store_mult_input_gate_partial_derivative;
	derivatives[neuron_derivatives_start + 16] = store_mult_candidate_gate_partial_derivative;

	data_t cell_addition_derivative = forget_out_mult_derivative + store_multiplication_derivative;
	data_t store_addition_multiplication_partial_derivative = forget_out_mult_derivative;
	data_t store_state_partial_derivative = store_multiplication_derivative;

	derivatives[neuron_derivatives_start + 17] = store_addition_multiplication_partial_derivative;
	derivatives[neuron_derivatives_start + 18] = store_state_partial_derivative;



	// Output gate
	data_t output_tanh = execution_values[neuron_execution_values_start + 8];
	data_t output_weight_multiplication = execution_values[neuron_execution_values_start + 9];

	data_t output_tanh_derivative = device_tanh_derivative(output_cell_state);
	derivatives[neuron_derivatives_start + 23] = output_tanh_derivative;

	data_t output_weight_derivative = sigmoid_lh + sigmoid_lh_derivative * output_weight;
	data_t output_weight_partial_derivative = sigmoid_lh;
	data_t output_sigmoid_partial_derivative = sigmoid_lh_derivative * output_weight;

	derivatives[neuron_derivatives_start + 19] = output_weight_partial_derivative;
	derivatives[neuron_derivatives_start + 20] = output_sigmoid_partial_derivative;

	data_t output_multiplication_derivative = output_tanh * output_weight_derivative + output_tanh_derivative * output_weight_multiplication;
	data_t output_multiplication_weight_mult_partial_derivative = output_tanh * output_weight_derivative;
	data_t output_multiplication_tanh_partial_derivative = output_weight_multiplication * output_tanh_derivative;

	derivatives[neuron_derivatives_start + 21] = output_multiplication_weight_mult_partial_derivative;
	derivatives[neuron_derivatives_start + 22] = output_multiplication_tanh_partial_derivative;

	data_t output_hidden_state_derivative = derivatives[neuron_derivatives_start + 3] =
			output_multiplication_derivative;

	data_t output_cell_state_derivative = derivatives[neuron_derivatives_start + 4] =
			cell_addition_derivative;
}
*/