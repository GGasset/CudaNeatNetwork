#include "NeatConnections.h"
#include "cuda_functionality.cuh"
#include <cstddef>

NeatConnections::NeatConnections(size_t previous_layer_start, size_t previous_layer_length, size_t neuron_count)
{
	connection_type = ConnectionTypes::NEAT;

	contains_irregular_connections = true;
	this->neuron_count = neuron_count;
	this->connection_count = neuron_count * previous_layer_length;
	cudaMalloc(&weights, sizeof(field_t) * connection_count);
	cudaMalloc(&biases, sizeof(field_t) * neuron_count);
	cudaMalloc(&connection_points, sizeof(size_t) * connection_count);
	cudaMalloc(&connection_neuron_i, sizeof(size_t) * connection_count);
	cudaDeviceSynchronize();

	generate_random_values(&weights, connection_count, 0, previous_layer_length);
	//cudaMemset(biases, 0, sizeof(field_t) * neuron_count);
	//generate_random_values(&biases, neuron_count, 0, neuron_count);

	//cudaMemset(weights, 0, sizeof(field_t) * connection_count);
	cudaMemset(biases, 0, sizeof(field_t) * neuron_count);
	cudaDeviceSynchronize();

	//add_to_array kernel (connection_count / 32 + (connection_count % 32 > 0), 32) (weights, connection_count, 1);
	add_to_array kernel (neuron_count / 32 + (neuron_count % 32 > 0), 32) (biases, neuron_count, 1);
	cudaDeviceSynchronize();
	
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
	gradient_hyperparameters hyperparameters, IOptimizer* optimizer
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
	unsigned int* device_count = 0;
	cudaMalloc(&device_count, sizeof(unsigned int));
	cudaDeviceSynchronize();

	cudaMemset(device_count, 0, sizeof(unsigned int));
	count_value kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (neuron_i, connection_neuron_i, connection_count, device_count);
	cudaDeviceSynchronize();

	unsigned int count = 0;
	cudaMemcpy(&count, device_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(device_count);
	return (size_t)count;
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

		size_t to_add_neuron_i = 0;
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

	connection_points = cuda_realloc(connection_points, connection_count, new_connection_count, true);
	cudaMemcpy(connection_points + connection_count, to_add_connection_points.data(), sizeof(size_t) * added_connection_count, cudaMemcpyHostToDevice);

	connection_neuron_i = cuda_realloc(connection_neuron_i, connection_count, new_connection_count, true);
	add_to_array kernel(added_connection_count / 32 + (added_connection_count % 32 > 0), 32) (
		connection_neuron_i + connection_count, added_connection_count, neuron_count
	);

	weights = cuda_realloc(weights, connection_count, new_connection_count, true);
	IConnections::generate_random_values(&weights, added_connection_count, connection_count, added_connection_count);
	biases = cuda_push_back(biases, neuron_count, (field_t)0, true);

	connection_count = new_connection_count;
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

	connection_points = cuda_realloc(connection_points, connection_count, new_connection_count, true);
	cudaMemcpy(connection_points + connection_count, added_neuron_i__added_connection_count_times.data(), sizeof(size_t) * added_connection_count, cudaMemcpyHostToDevice);

	connection_neuron_i = cuda_realloc(connection_neuron_i, connection_count, new_connection_count, true);
	cudaMemcpy(connection_neuron_i + connection_count, added_connections_neuron_i->data(), sizeof(size_t) * added_connection_count, cudaMemcpyHostToDevice);

	weights = cuda_realloc(weights, connection_count, new_connection_count, true);
	generate_random_values(&weights, added_connection_count, connection_count, new_connection_count);

	connection_count = new_connection_count;
}

void NeatConnections::remove_neuron(size_t neuron_i)
{
	size_t to_delete_connection_count = get_connection_count_at(neuron_i);

	connection_points = cuda_remove_occurrences(connection_neuron_i, neuron_i, connection_points, connection_count, true);
	weights = cuda_remove_occurrences(connection_neuron_i, neuron_i, weights, connection_count, true);
	
	connection_neuron_i = cuda_remove_occurrences(connection_neuron_i, neuron_i, connection_neuron_i, connection_count, true);
	connection_neuron_i = cuda_add_to_occurrences(connection_neuron_i, size_t_bigger_than_compare_func, neuron_i, connection_neuron_i, (size_t)-1, connection_count - to_delete_connection_count, true);
	biases = cuda_remove_elements(biases, neuron_count, neuron_i, 1, true);

	connection_count -= to_delete_connection_count;
	cudaDeviceSynchronize();
}

void NeatConnections::adjust_to_removed_neuron(size_t neuron_i, std::vector<size_t>* removed_connections_neuron_i)
{
	size_t to_delete_connection_count = get_connection_count_at(neuron_i);
	if (!to_delete_connection_count) return;

	size_t *host_connection_points = new size_t[connection_count];
	cudaMemcpy(host_connection_points, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);

	size_t *host_connection_neuron_i = new size_t[connection_count];
	cudaMemcpy(host_connection_neuron_i, connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < connection_count; i++)
		if (host_connection_points[i] == neuron_i)
			removed_connections_neuron_i->push_back(host_connection_neuron_i[i]);

	delete[] host_connection_points;
	delete[] host_connection_neuron_i;

	weights = cuda_remove_occurrences(connection_points, neuron_i, weights, connection_count, true);
	connection_neuron_i = cuda_remove_occurrences(connection_points, neuron_i, connection_neuron_i, connection_count, true);
	connection_points = cuda_remove_occurrences(connection_points, neuron_i, connection_points, connection_count, true);

	connection_count -= to_delete_connection_count;
}

IConnections* NeatConnections::connections_specific_clone()
{
	NeatConnections* connections = new NeatConnections();
	cudaMalloc(&connections->connection_points, sizeof(size_t) * connection_count);
	cudaMalloc(&connections->connection_neuron_i, sizeof(size_t) * connection_count);
	cudaDeviceSynchronize();
	cudaMemcpy(connections->connection_points, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(connections->connection_neuron_i, connection_neuron_i, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
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
}

void NeatConnections::specific_deallocate()
{
	cudaFree(connection_points);
	cudaFree(connection_neuron_i);
}

