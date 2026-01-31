#ifndef DENSE_CONNECTIONS
#define DENSE_CONNETIONS

#include "DenseConnections.h"
#include "stdio.h"

DenseConnections::DenseConnections(
	size_t previous_layer_activations_start, size_t previous_layer_length, size_t neuron_count,
	initialization_parameters weights_init, initialization_parameters bias_init
)
{
	connection_type = ConnectionTypes::Dense;

	this->neuron_count = neuron_count;
	this->connection_count = previous_layer_length * neuron_count;
	this->previous_layer_activations_start = previous_layer_activations_start;
	this->previous_layer_length = previous_layer_length;

	initialize_parameters(&weights, connection_count, weights_init);
	initialize_parameters(&biases, neuron_count, bias_init);
}

DenseConnections::DenseConnections()
{
	connection_type = ConnectionTypes::Dense;
}

void DenseConnections::plinear_function(
	size_t t_count, data_t *activations, data_t *execution_vals, layer_properties properties, nn_lens lengths,
	size_t gaps_between_usable_arrays_t_count
)
{
	size_t total_inputs = t_count * connection_count;
	data_t *extracted_activations = 0;
	cudaMalloc(&extracted_activations, sizeof(data_t) * total_inputs);
	extract_activations_dense n_threads(total_inputs) (
		t_count, activations, lengths.neurons, neuron_count,
		extracted_activations,
		previous_layer_activations_start, previous_layer_length, gaps_between_usable_arrays_t_count
	);
	cudaDeviceSynchronize();

	repetitive_element_wise_multiply n_threads(total_inputs) (
		extracted_activations, total_inputs, weights, connection_count,
		extracted_activations
	);
	cudaDeviceSynchronize();

	size_t n_linear_funcs = t_count * neuron_count;
	data_t *pondered_sum = multi_PRAM_add(extracted_activations, previous_layer_length, n_linear_funcs);
	repetitive_sum n_threads(total_inputs) (
		pondered_sum, n_linear_funcs, biases, neuron_count, pondered_sum
	);
	cudaDeviceSynchronize();
	cudaFree(extracted_activations);

	// Insert linear functions
	insert_execution_values n_threads(n_linear_funcs) (
		t_count, lengths.execution_values, neuron_count, 
		properties.execution_values_start, properties.execution_values_per_neuron, 0,
		gaps_between_usable_arrays_t_count,
		pondered_sum, execution_vals
	);
	cudaDeviceSynchronize();
	cudaFree(pondered_sum);
}

void DenseConnections::pbackpropagate(
	size_t t_count, nn_lens lengths, layer_properties props,
	data_t *activations, data_t *grads, data_t *costs, 
	size_t gaps_between_usable_arrays_t_count
)
{
	size_t total_connections = connection_count * t_count;
	size_t n_linear_funcs = neuron_count * t_count;

	data_t *previous_layer_activations = 0;
	cudaMalloc(&previous_layer_activations, sizeof(data_t) * total_connections);

	data_t *bias_gradients = 0;
	cudaMalloc(&bias_gradients, sizeof(data_t) * n_linear_funcs);

	network_value_extract n_threads(total_connections) (
		t_count, lengths.neurons, neuron_count, 1, gaps_between_usable_arrays_t_count,
		props.activations_start, 0, 1,
		activations, previous_layer_activations
	);

	network_value_extract n_threads(n_linear_funcs) (
		t_count, lengths.gradients, neuron_count, previous_layer_length + 1 + props.gradients_per_neuron,
		gaps_between_usable_arrays_t_count, props.gradients_start, 0, 1,
		grads, bias_gradients
	);
	cudaDeviceSynchronize();

	// Expand bias gradients (repeat each one its connection_count times)
	data_t *weight_gradients = 0;
	cudaMalloc(&weight_gradients, sizeof(data_t) * total_connections);
	clone_arr_values_n_times n_threads(total_connections) (
		weight_gradients, n_linear_funcs, previous_layer_length, weight_gradients, total_connections
	);
	cudaDeviceSynchronize();
	cudaFree(bias_gradients);

	// Clone bias grads (one for activation costs, one for weight costs)

	data_t *activations_gradients = cuda_clone_arr(weight_gradients, total_connections);

	// Multiply bias grads with activations using element wise

	element_wise_multiply n_threads(total_connections) (
		weight_gradients, previous_layer_activations, total_connections
	);

	// Multiply bias grads copy with weights using repetitive element wise
	repetitive_element_wise_multiply n_threads(total_connections) (
		activations_gradients, total_connections, weights, connection_count, activations_gradients
	);
	cudaDeviceSynchronize();

	multiply_array n_threads(total_connections) (activations_gradients, total_connections, -1);
	cudaDeviceSynchronize();

	cudaFree(previous_layer_activations);

	// Insert gradients
	network_value_insert n_threads(total_connections) (
		t_count, lengths.neurons, neuron_count, 1, gaps_between_usable_arrays_t_count, props.activations_start, 0, 1,
		activations_gradients, costs
	);
	network_value_insert n_threads(total_connections) (
		t_count, lengths.gradients, neuron_count, previous_layer_length + 1 + props.gradients_per_neuron, gaps_between_usable_arrays_t_count,
		props.gradients_start, 1, previous_layer_length, weight_gradients, grads
	);
	cudaDeviceSynchronize();

	cudaFree(activations_gradients);
	cudaFree(weight_gradients);
}

void DenseConnections::pget_derivative(
	size_t t_count, data_t *activations, data_t *derivatives, size_t gaps_between_usable_arrays_t_count, 
	layer_properties props, nn_lens lengths
)
{

	size_t total_prev_layer_activations = previous_layer_length * t_count;
	data_t *previous_layer_activations = 0;
	cudaMalloc(&previous_layer_activations, sizeof(data_t) * total_prev_layer_activations);
	network_value_extract n_threads(total_prev_layer_activations) (
		t_count, lengths.derivative, previous_layer_length, 1, gaps_between_usable_arrays_t_count,
		previous_layer_activations_start, 0, 1,
		activations, previous_layer_activations
	);
	cudaDeviceSynchronize();

	size_t total_connection_count = connection_count * t_count;
	data_t *connections_activation = 0;
	cudaMalloc(&connections_activation, sizeof(data_t) * total_connection_count);
	repetitive_copy n_threads(total_connection_count) (
		connections_activation, total_connection_count, previous_layer_activations, total_prev_layer_activations
	);
	cudaDeviceSynchronize();

	repetitive_element_wise_multiply n_threads(total_connection_count) (
		connections_activation, total_connection_count, weights, connection_count, connections_activation
	);

	cudaFree(previous_layer_activations);
	cudaDeviceSynchronize();

	data_t *out_derivatives = multi_PRAM_add(connections_activation, connection_count, t_count);
	cudaFree(connections_activation);

	size_t total_layer_length = neuron_count * t_count;
	network_value_insert n_threads(total_layer_length) (
		t_count, lengths.derivative, neuron_count, props.derivatives_per_neuron, gaps_between_usable_arrays_t_count,
		props.derivatives_start, 0, 1,
		out_derivatives, derivatives
	);
	cudaDeviceSynchronize();
}

void DenseConnections::linear_function(size_t activations_start, data_t* activations,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
)
{
	if (connection_count > PRAM_THRESHOLD)
	{
		data_t *input_arr = 0;
		cudaMalloc(&input_arr, sizeof(data_t) * connection_count);
		repetitive_copy n_threads(connection_count) (
			input_arr, connection_count, 
			activations + activations_start + previous_layer_activations_start, previous_layer_length
		);
		cudaDeviceSynchronize();

		element_wise_multiply n_threads(connection_count) (input_arr, weights, connection_count);
		cudaDeviceSynchronize();

		data_t *linear_funcs = multi_PRAM_add(input_arr, previous_layer_length, neuron_count);
		cudaFree(input_arr);

		set_execution_values n_threads(neuron_count) (
			execution_values + execution_values_start + execution_values_layer_start, linear_funcs,
			layer_execution_values_per_neuron, 0, neuron_count
		);
		cudaFree(linear_funcs);
		cudaDeviceSynchronize();
	}
	else
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
	gradient_hyperparameters hyperparameters, Optimizers optimizer
)
{
	cud_dense_gradient_subtraction kernel(dim3(previous_layer_length / 32 + (previous_layer_length % 32 > 0), neuron_count), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		weights, previous_layer_length, neuron_count, hyperparameters, optimizer
	);
	bias_gradient_subtraction kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		biases, neuron_count, hyperparameters, optimizer
	);
	cudaDeviceSynchronize();
}

IConnections* DenseConnections::connections_specific_clone()
{
	DenseConnections* connections = new DenseConnections();
	connections->previous_layer_activations_start = previous_layer_activations_start;
	connections->previous_layer_length = previous_layer_length;
	return connections;
}

void DenseConnections::specific_save(FILE* file)
{
	fwrite(&previous_layer_activations_start, sizeof(size_t), 1, file);
	fwrite(&previous_layer_length, sizeof(size_t), 1, file);
}

void DenseConnections::load(FILE* file)
{
	load_neuron_metadata(file);

	fread(&previous_layer_activations_start, sizeof(size_t), 1, file);
	fread(&previous_layer_length, sizeof(size_t), 1, file);

	load_IConnections_data(file);
}

size_t DenseConnections::get_connection_count_at(size_t neuron_i)
{
	return previous_layer_length;
}
#endif
