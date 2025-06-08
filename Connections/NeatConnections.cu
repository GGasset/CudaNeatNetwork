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

	size_t added_connection_count = 0;
	size_t new_connection_count = connection_count;
	for (size_t i = 0; i < previous_layer_length; i++)
	{
		bool added = previous_layer_connection_probability > get_random_float();
		if (!added) continue;

		added_connection_point_layer_neuron_i.insert(i);

		connection_points = cuda_push_back(connection_points, new_connection_count, previous_layer_activations_start + i, true);
		connection_neuron_i = cuda_push_back(connection_neuron_i, new_connection_count, neuron_count, true);
		weights = cuda_push_back(weights, new_connection_count, (field_t)get_random_float(), true);

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

			connection_neuron_i = cuda_push_back(connection_neuron_i, new_connection_count, neuron_count, true);
			connection_points = cuda_push_back(connection_points, new_connection_count, previous_layer_activations_start + i, true);
			weights = cuda_push_back(weights, new_connection_count, (field_t)get_random_float(), true);

			added_connection_count++;
			new_connection_count++;
			added_connection_point_layer_neuron_i.insert(i);
			i = previous_layer_length;
		}
	}
	multiply_array kernel(added_connection_count / 32 + (added_connection_count % 32 > 0), 32) (
		weights + connection_count, added_connection_count, 1.0 / added_connection_count
	);
	biases = cuda_push_back(biases, neuron_count, (field_t)0, true);
	cudaDeviceSynchronize();
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
	size_t connection_count_until_deletion = 0;
	for (size_t i = 0; i < neuron_i; i++)
		connection_count_until_deletion += get_connection_count_at(i);

	size_t connection_count_after_deletion = 0;
	for (size_t i = neuron_i + 1; i < neuron_count; i++)
		connection_count_after_deletion += get_connection_count_at(i);

	size_t to_delete_connection_count = get_connection_count_at(neuron_i);
	
	size_t connections_remove_start = connection_count_until_deletion;// - (connection_count_until_deletion != 0);
	size_t new_connection_count = connection_count - to_delete_connection_count;

	connection_neuron_i = cuda_remove_elements(connection_neuron_i, connection_count, connections_remove_start, to_delete_connection_count, true);
	add_to_array kernel(new_connection_count / 32 + (new_connection_count % 32 > 0), 32) (
		connection_neuron_i + connection_count_until_deletion, new_connection_count - connection_count_until_deletion, -1
	);

	connection_points = cuda_remove_elements(connection_points, connection_count, connections_remove_start, to_delete_connection_count, true);
	weights = cuda_remove_elements(weights, connection_count, connections_remove_start, to_delete_connection_count, true);
	biases = cuda_remove_elements(biases, neuron_count, neuron_i, 1, true);

	connection_count -= to_delete_connection_count;
	cudaDeviceSynchronize();
}

void NeatConnections::adjust_to_removed_neuron(size_t neuron_i, std::vector<size_t>* removed_connections_neuron_i)
{
	size_t* host_connection_points = new size_t[connection_count];
  size_t* host_connection_neuron_i = new size_t[connection_count];
	field_t* host_weights = new field_t[connection_count];

	cudaMemcpy(host_connection_points, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_weights, weights, sizeof(field_t) * connection_count, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_connection_neuron_i, connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	auto connection_points_vector = std::vector<size_t>();
	auto vector_weights = std::vector<field_t>();
  std::vector<size_t> vector_connection_neuron_i;
	for (size_t i = 0; i < connection_count; i++)
	{
		// Adjust connections for index change while transforming points to a vector
		connection_points_vector.push_back(host_connection_points[i]);
		vector_weights.push_back(host_weights[i]);
    vector_connection_neuron_i.push_back(host_connection_neuron_i[i]);
	}

  delete[] host_connection_points;
  delete[] host_connection_neuron_i;
  delete[] host_weights;

	size_t found_i = 0;
	while (true)
	{
		// Search for connections pointing to neuron_i, break if not found
		uint8_t found = false;
		for (size_t i = found_i; i < connection_count && !found; i++)
		{
			found = connection_points_vector[i] == neuron_i;
			found_i = i;
		}
		if (!found)
			break;

		// Get the neuron containing the connection
		size_t connection_neuron_i = vector_connection_neuron_i[found_i];

		// Update info
		removed_connections_neuron_i->push_back(connection_neuron_i);
		vector_weights.erase(vector_weights.begin() + found_i);
		connection_points_vector.erase(connection_points_vector.begin() + found_i);
    vector_connection_neuron_i.erase(vector_connection_neuron_i.begin() + found_i);
		connection_count--;
	}
	for (size_t i = 0; i < connection_count; i++)
	{
		connection_points_vector[i] -= connection_points_vector[i] > neuron_i;
	}
	cudaFree(connection_points);
	cudaFree(weights);
  cudaFree(connection_neuron_i);
	cudaDeviceSynchronize();

	cudaMalloc(&connection_points, sizeof(size_t) * connection_count);
	cudaMalloc(&weights, sizeof(field_t) * connection_count);
  cudaMalloc(&connection_neuron_i, sizeof(size_t) * connection_count);
	cudaDeviceSynchronize();

	cudaMemcpy(connection_points, connection_points_vector.data(), sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaMemcpy(weights, vector_weights.data(), sizeof(field_t) * connection_count, cudaMemcpyHostToDevice);
  cudaMemcpy(connection_neuron_i, vector_connection_neuron_i.data(), sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
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
	size_t *host_connection_points, *host_connection_neuron_i;
	host_connection_points = new size_t[connection_count];
	host_connection_neuron_i = new size_t[connection_count];

	cudaMemcpy(host_connection_points, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_connection_neuron_i, connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	fwrite(host_connection_points, sizeof(size_t), connection_count, file);
	fwrite(host_connection_neuron_i, sizeof(size_t), connection_count, file);

	delete[] host_connection_points;
	delete[] host_connection_neuron_i;
}

void NeatConnections::load(FILE* file)
{
	load_neuron_metadata(file);

	size_t *host_connection_points = 0;
	size_t *host_connection_neuron_i = 0;

	host_connection_points = new size_t[connection_count];
	host_connection_neuron_i = new size_t[connection_count];

	fread(host_connection_points, sizeof(size_t), connection_count, file);
	fread(host_connection_neuron_i, sizeof(size_t), connection_count, file);

	cudaMalloc(&connection_points, sizeof(size_t) * connection_count);
	cudaMalloc(&connection_neuron_i, sizeof(size_t) * connection_count);
	cudaDeviceSynchronize();
	
	cudaMemcpy(connection_points, host_connection_points, sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaMemcpy(connection_neuron_i, host_connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	delete[] host_connection_points;
	delete[] host_connection_neuron_i;

	load_IConnections_data(file);
}

void NeatConnections::specific_deallocate()
{
	cudaFree(connection_points);
	cudaFree(connection_neuron_i);
}

