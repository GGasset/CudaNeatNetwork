#include "NeatConnections.h"
#include "cuda_functionality.cuh"
#include <cstddef>

NeatConnections::NeatConnections(
	size_t previous_layer_start, size_t previous_layer_length, size_t neuron_count,
	initialization_parameters weights_init, initialization_parameters bias_init
)
{
	connection_type = ConnectionTypes::NEAT;

	contains_irregular_connections = true;
	this->neuron_count = neuron_count;
	this->connection_count = neuron_count * previous_layer_length;
	cudaMalloc(&connection_points, sizeof(size_t) * connection_count);
	cudaMalloc(&connection_neuron_i, sizeof(size_t) * connection_count);

	initialize_parameters(&biases, connection_count, weights_init);
	initialize_parameters(&weights, connection_count, bias_init);

	max_connections_at_layer = previous_layer_length;
	
	size_t* host_connection_points = new size_t[connection_count];
	size_t* host_connection_neuron_i = new size_t[connection_count];
	//connection_counts = new size_t[neuron_count];
	for (size_t i = 0; i < neuron_count; i++)
	{
		for (size_t j = 0; j < previous_layer_length; j++)
		{
			host_connection_points[i * previous_layer_length + j] = previous_layer_start + j;
			host_connection_neuron_i[i * previous_layer_length + j] = i;
		}
		//connection_counts[i] = previous_layer_length;
	}
	cudaMemcpy(connection_points, host_connection_points, sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaMemcpy(connection_neuron_i, host_connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	delete[] host_connection_neuron_i;
	delete[] host_connection_points;
}

NeatConnections::NeatConnections()
{
	connection_type = ConnectionTypes::NEAT;
}

void NeatConnections::plinear_function(
	size_t t_count, data_t *activations, data_t *execution_vals, layer_properties properties, nn_lens lengths, 
	size_t gaps_between_usable_arrays_t_count
)
{
	size_t n_linear_funcs = neuron_count * t_count;
	size_t extracted_activation_count = max_connections_at_layer * n_linear_funcs;

	data_t *extracted_activations = 0;
	cudaMalloc(&extracted_activations, sizeof(data_t) * extracted_activation_count);
	get_pondered_activations_neat n_threads(extracted_activation_count) (
		t_count, activations, lengths.neurons, neuron_count, connection_count,
		extracted_activations, connection_points, connection_neuron_i, weights, max_connections_at_layer,
		gaps_between_usable_arrays_t_count
	);
	cudaDeviceSynchronize();

	data_t *linear_funcs = multi_PRAM_add(extracted_activations, max_connections_at_layer, n_linear_funcs);
	cudaFree(extracted_activations);

	repetitive_sum n_threads(n_linear_funcs) (linear_funcs, n_linear_funcs, biases, neuron_count, linear_funcs);
	cudaDeviceSynchronize();

	insert_execution_values n_threads(n_linear_funcs) (
		t_count, lengths.execution_values, neuron_count, 
		properties.execution_values_start, properties.execution_values_per_neuron,
		0, gaps_between_usable_arrays_t_count, 
		linear_funcs, execution_vals
	);
	cudaDeviceSynchronize();
}

void NeatConnections::pbackpropagate(
	size_t t_count, nn_lens lengths, layer_properties props,
	data_t *activations, data_t *grads, data_t *costs,
	size_t gaps_between_usable_arrays_t_count
)
{
	NEAT_backpropagate n_threads(t_count * connection_count) (
		t_count, activations, grads, costs, weights, connection_points, connection_neuron_i, connection_count,
		lengths, props, gaps_between_usable_arrays_t_count
	);
	cudaDeviceSynchronize();
}

void NeatConnections::pget_derivative(
	size_t t_count, data_t *activations, data_t *derivatives, size_t gaps_between_usable_arrays_t_count,
	layer_properties props, nn_lens lengths
)
{
	size_t total_connection_count = connection_count * t_count;
	size_t raw_conn_derivative_count =  max_connections_at_layer * neuron_count * t_count;

	data_t *conn_derivatives = 0;
	cudaMalloc(&conn_derivatives, sizeof(data_t) * raw_conn_derivative_count);
	NEAT_unsummed_linear_func_derivative n_threads(raw_conn_derivative_count) (
		t_count, gaps_between_usable_arrays_t_count, neuron_count,
		connection_points, connection_neuron_i, max_connections_at_layer, connection_count,
		lengths, props, weights, activations, conn_derivatives
	);
	cudaDeviceSynchronize();
	
	size_t total_layer_neuron_count = neuron_count * t_count;
	data_t *calc_derivatives = multi_PRAM_add(conn_derivatives, max_connections_at_layer, total_layer_neuron_count);
	add_to_array n_threads(total_layer_neuron_count) (
		calc_derivatives, total_layer_neuron_count, 1
	);
	cudaDeviceSynchronize();

	network_value_insert n_threads(total_layer_neuron_count) (
		t_count, lengths.derivative, neuron_count, props.derivatives_per_neuron, gaps_between_usable_arrays_t_count,
		props.derivatives_start, 0, 1, 
		calc_derivatives, derivatives
	);
	cudaDeviceSynchronize();
}

void NeatConnections::linear_function(
	size_t activations_start, data_t* activations, 
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
)
{
	cud_NEAT_linear_function kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
		connection_count, weights, connection_points, connection_neuron_i,
		activations_start, activations,
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cud_add_biases kernel(dim3(neuron_count / 32 + (neuron_count % 32 > 0), 1, 1), 32) (
		neuron_count, biases, 
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cudaDeviceSynchronize();
}

void NeatConnections::calculate_derivative(
	size_t activations_start, data_t* activations, 
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
)
{
	cud_NEAT_linear_function_derivative kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
		activations_start, activations,
		derivatives_start, derivatives_layer_start, derivatives_per_neuron, derivatives,
		connection_count, weights, connection_points, connection_neuron_i
	);

	cud_add_bias_derivative kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		neuron_count, derivatives_start, derivatives_layer_start, derivatives_per_neuron, derivatives
	);
	cudaDeviceSynchronize();
}

void NeatConnections::calculate_gradients(
	data_t* activations, size_t activations_start, 
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, 
	data_t* costs, size_t costs_start
)
{
	cud_NEAT_gradient_calculation kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
		activations, activations_start,
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start,
		connection_count, weights, connection_points, connection_neuron_i
	);
	cudaDeviceSynchronize();
}

void NeatConnections::subtract_gradients(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, 
	gradient_hyperparameters hyperparameters, Optimizers optimizer
)
{
	cud_NEAT_gradient_subtraction kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		connection_neuron_i, connection_count, neuron_count,
		weights,
		hyperparameters, optimizer
	);
	bias_gradient_subtraction kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		biases, neuron_count, hyperparameters, optimizer
	);
	cudaDeviceSynchronize();
}

size_t NeatConnections::get_connection_count_at(size_t neuron_i)
{
	return count_value(neuron_i, connection_neuron_i, connection_count);
}

void NeatConnections::set_max_connections_at_layer()
{
	size_t max = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t connection_count_at_i = get_connection_count_at(i);
		max += (connection_count_at_i - (long)max) * (connection_count_at_i > max);
	}
	max_connections_at_layer = max;
}

void NeatConnections::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
	std::set<size_t> added_connection_point_layer_neuron_i = std::set<size_t>();
	std::vector<size_t> to_add_connection_points = std::vector<size_t>();


	size_t added_connection_count = 0;
	size_t new_connection_count = connection_count;
	for (size_t i = 0; i < previous_layer_length; i++)
	{
		bool added = previous_layer_connection_probability > get_random_float();
		if (!added) continue;

		added_connection_point_layer_neuron_i.insert(i);
		to_add_connection_points.push_back(previous_layer_activations_start + i);

		added_connection_count++;
		new_connection_count++;
	}

	while (added_connection_count < min_connections && added_connection_count < previous_layer_length)
	{
		size_t remaining_connection_count = previous_layer_length - added_connection_count;
		size_t remaining_connection_i_to_add = rand() % remaining_connection_count;

		for (size_t i = 0, counter = 0; i < previous_layer_length; i++)
		{
			counter += (added_connection_point_layer_neuron_i.count(i) == 0);
			if (counter != remaining_connection_i_to_add + 1) continue;

			to_add_connection_points.push_back(previous_layer_activations_start + i);

			added_connection_count++;
			new_connection_count++;
			added_connection_point_layer_neuron_i.insert(i);
			i = previous_layer_length;
		}
	}

	connection_points = cuda_append_array(
		connection_points, connection_count, to_add_connection_points.data(), added_connection_count,
		true, true
	);

	// Insert the same indice multiple times at the end of connection_neuron_i
	connection_neuron_i = cuda_realloc(connection_neuron_i, connection_count, new_connection_count, true);
	add_to_array kernel(added_connection_count / 32 + (added_connection_count % 32 > 0), 32) (
		connection_neuron_i + connection_count, added_connection_count, neuron_count
	);

	weights = cuda_realloc(weights, connection_count, new_connection_count, true);
	generate_random_values(weights, added_connection_count, connection_count, added_connection_count, true);
	biases = cuda_push_back(biases, neuron_count, (field_t)0, true);

	connection_count = new_connection_count;

	set_max_connections_at_layer();
}

void NeatConnections::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability, std::vector<size_t>* added_connections_neuron_i)
{
	std::vector<size_t> added_neuron_i__added_connection_count_times = std::vector<size_t>();
	size_t added_connection_count = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		bool added = connection_probability >= get_random_float();
		added_connection_count += added;

		if (added)
		{
			added_neuron_i__added_connection_count_times.push_back(added_neuron_i);

			added_connections_neuron_i->push_back(i);
		}
	}
	if (!added_connection_count) return;

	size_t new_connection_count = connection_count + added_connection_count;

	connection_points = cuda_append_array(
		connection_points, connection_count,
		added_neuron_i__added_connection_count_times.data(), added_connection_count, true, true
	);

	connection_neuron_i = cuda_append_array(
		connection_neuron_i, connection_count, added_connections_neuron_i->data(), added_connection_count, true, true
	);

	weights = cuda_realloc(weights, connection_count, new_connection_count, true);
	generate_random_values(weights, added_connection_count, connection_count, new_connection_count, true);

	connection_count = new_connection_count;

	set_max_connections_at_layer();
}

void NeatConnections::remove_neuron(size_t neuron_i, std::vector<size_t> *removed_connections_i)
{
	size_t *host_connection_neuron_i = new size_t[connection_count];
	cudaMemcpy(host_connection_neuron_i, connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < connection_count && host_connection_neuron_i && removed_connections_i; i++)
		if (host_connection_neuron_i[i] == neuron_i)
			removed_connections_i->push_back(i);

	delete[] host_connection_neuron_i;

	size_t to_delete_connection_count = get_connection_count_at(neuron_i);

	connection_points = cuda_remove_occurrences(connection_neuron_i, neuron_i, connection_points, connection_count, true);
	weights = cuda_remove_occurrences(connection_neuron_i, neuron_i, weights, connection_count, true);
	
	connection_neuron_i = cuda_remove_occurrences(connection_neuron_i, neuron_i, connection_neuron_i, connection_count, true);
	connection_neuron_i = cuda_add_to_occurrences(connection_neuron_i, size_t_bigger_than_compare_func, neuron_i, connection_neuron_i, (size_t)-1, connection_count - to_delete_connection_count, true);
	biases = cuda_remove_elements(biases, neuron_count, neuron_i, 1, true);

	connection_count -= to_delete_connection_count;
	set_max_connections_at_layer();
}

void NeatConnections::adjust_to_removed_neuron(size_t neuron_i, std::vector<size_t>* removed_connections_neuron_i, std::vector<size_t>* removed_connections_i)
{
	size_t to_delete_connection_count = get_connection_count_at(neuron_i);
	if (!to_delete_connection_count) return;

	size_t *host_connection_points = new size_t[connection_count];
	cudaMemcpy(host_connection_points, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);

	size_t *host_connection_neuron_i = new size_t[connection_count];
	cudaMemcpy(host_connection_neuron_i, connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < connection_count && removed_connections_neuron_i && removed_connections_i; i++)
		if (host_connection_points[i] == neuron_i)
		{
			removed_connections_i->push_back(i);
			removed_connections_neuron_i->push_back(host_connection_neuron_i[i]);
		}

	delete[] host_connection_points;
	delete[] host_connection_neuron_i;

	weights = cuda_remove_occurrences(connection_points, neuron_i, weights, connection_count, true);
	connection_neuron_i = cuda_remove_occurrences(connection_points, neuron_i, connection_neuron_i, connection_count, true);
	connection_points = cuda_remove_occurrences(connection_points, neuron_i, connection_points, connection_count, true);

	connection_count -= to_delete_connection_count;
	set_max_connections_at_layer();
}

IConnections* NeatConnections::connections_specific_clone()
{
	NeatConnections* connections = new NeatConnections();
	connections->connection_points = cuda_clone_arr(connection_points, connection_count);
	connections->connection_neuron_i = cuda_clone_arr(connection_neuron_i, connection_count);
	connections->max_connections_at_layer = max_connections_at_layer;
	return connections;
}

void NeatConnections::specific_save(FILE* file)
{
	save_array(connection_points, connection_count, file, true);
	save_array(connection_neuron_i, connection_count, file, true);
}

void NeatConnections::load(FILE* file)
{
	load_neuron_metadata(file);

	connection_points = load_array<size_t>(connection_count, file, true);
	connection_neuron_i = load_array<size_t>(connection_count, file, true);

	load_IConnections_data(file);
	set_max_connections_at_layer();
}

void NeatConnections::specific_deallocate()
{
	cudaFree(connection_points);
	cudaFree(connection_neuron_i);
}

