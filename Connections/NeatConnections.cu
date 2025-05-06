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
	/*size_t connections_start = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t connection_count = connection_counts[i];
		cud_NEAT_gradient_calculation kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
			activations, activations_start,
			gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
			costs, costs_start,
			i, connection_count, weights, connection_points, connections_start
		);

		connections_start += connection_count;
	}*/
	cudaDeviceSynchronize();
}

void NeatConnections::subtract_gradients(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, 
	data_t learning_rate, data_t gradient_clip
)
{
	/*size_t connections_start = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t connection_count = connection_counts[i];
		cud_NEAT_gradient_subtraction kernel(connection_count / 32 + (connection_count % 32 + 1), 32) (
			gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
			i, connection_count, weights, connections_start,
			learning_rate, dropout, gradient_clip
		);
		connections_start += connection_count;
	}*/
	cud_NEAT_gradient_subtraction kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		connection_neuron_i, connection_count,
		weights,
		learning_rate, gradient_clip
	);
	bias_gradient_subtraction kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		biases, neuron_count, learning_rate, gradient_clip
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
	size_t added_connection_count = 0;
	size_t* device_tmp_connections = 0;
	field_t* device_tmp_biases = 0;
	field_t* device_tmp_weights = 0;
	size_t* tmp_connections_neuron_i = 0;
  auto tmp_connections = std::vector<size_t>();
	auto sampling_vector = std::vector<size_t>();
	for (size_t i = 0; i < previous_layer_length; i++)
	{
		if (rand() % 100000 / 100000.0 < previous_layer_connection_probability)
		{
			added_connection_count++;
			tmp_connections.push_back(i + previous_layer_activations_start);
			continue;
		}
		sampling_vector.push_back(i + previous_layer_activations_start);
	}
	while (tmp_connections.size() < min_connections && sampling_vector.size())
	{
		size_t i = rand() % sampling_vector.size();
		tmp_connections.push_back(sampling_vector.at(i));
		sampling_vector.erase(sampling_vector.begin() + i);
		added_connection_count++;
	}
	
	cudaMalloc(&device_tmp_connections, sizeof(size_t) * (connection_count + added_connection_count));
	cudaMalloc(&device_tmp_weights, sizeof(field_t) * (connection_count + added_connection_count));
	cudaMalloc(&device_tmp_biases, sizeof(field_t) * (neuron_count + 1));
  cudaMalloc(&tmp_connections_neuron_i , sizeof(size_t) * (connection_count + added_connection_count));
	cudaDeviceSynchronize();

	cudaMemcpy(device_tmp_weights, weights, sizeof(field_t) * connection_count, cudaMemcpyDeviceToDevice);
  
  cudaMemcpy(tmp_connections_neuron_i, connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyDeviceToDevice);
  cudaMemset(tmp_connections_neuron_i + connection_count, 0, sizeof(size_t) * added_connection_count);

	cudaMemcpy(device_tmp_biases, biases, sizeof(field_t) * neuron_count, cudaMemcpyDeviceToDevice);


	cudaMemcpy(device_tmp_connections, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_tmp_connections + connection_count, tmp_connections.data(), sizeof(size_t) * added_connection_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

  add_to_array kernel(added_connection_count / 32 + (added_connection_count % 32 > 0), 32) (tmp_connections_neuron_i + connection_count, added_connection_count, neuron_count);
	generate_random_values(&device_tmp_biases, 1, neuron_count, 1);
	generate_random_values(&device_tmp_weights, added_connection_count, connection_count, 1);

	cudaFree(weights);
	cudaFree(biases);
	cudaFree(connection_points);
	cudaDeviceSynchronize();
	
  
	/*size_t* tmp_connection_counts = new size_t[neuron_count + 1];
	cudaMemcpy(tmp_connection_counts, connection_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToHost);
	cudaDeviceSynchronize();
	tmp_connection_counts[neuron_count] = added_connection_count;
	delete[] connection_counts;*/

	//connection_counts = tmp_connection_counts;
	connection_neuron_i = tmp_connections_neuron_i;
  weights = device_tmp_weights;
	biases = device_tmp_biases;
	connection_points = device_tmp_connections;
	connection_count += added_connection_count;
}

void NeatConnections::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability, std::vector<size_t>* added_connections_neuron_i)
{
	// Transform data to a vector
	size_t *host_connection_points = new size_t[connection_count];
  size_t *host_connection_neuron_i = new size_t[connection_count];
	field_t* host_weights = new field_t[connection_count];
	
	cudaMemcpy(host_connection_points, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_weights, weights, sizeof(field_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_connection_neuron_i, connection_neuron_i, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	auto vector_connection_points = std::vector<size_t>();
	auto vector_weights = std::vector<field_t>();
  std::vector<size_t> vector_connection_neuron_i;
	for (size_t i = 0; i < connection_count; i++)
	{
		// Adjust connections for index change while transforming points to a vector
		vector_connection_points.push_back(host_connection_points[i] + (host_connection_points[i] >= added_neuron_i));
		vector_weights.push_back(host_weights[i]);
    vector_connection_neuron_i.push_back(host_connection_neuron_i[i]);
	}

	// Add connections
	size_t tmp_connection_count = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		uint8_t is_connection_added = rand() % 100000 / 100000.0 < connection_probability;
		size_t old_neuron_connection_count = get_connection_count_at(i);
		size_t new_neuron_connection_count = old_neuron_connection_count + is_connection_added;
		//connection_counts[i] = new_neuron_connection_count;
		
		if (is_connection_added)
		{
			added_connections_neuron_i->push_back(i);
			vector_connection_points.insert(vector_connection_points.begin() + tmp_connection_count + old_neuron_connection_count, added_neuron_i);
			vector_weights.insert(vector_weights.begin() + tmp_connection_count + old_neuron_connection_count, rand() % 100000 / 100000.0);
      vector_connection_neuron_i.insert(vector_connection_neuron_i.begin() + tmp_connection_count + old_neuron_connection_count, i);
		}

		connection_count += is_connection_added;
		tmp_connection_count += new_neuron_connection_count;
	}

	// Copy data to device, free old arrays and free host arrays
  cudaFree(connection_neuron_i);
	cudaFree(connection_points);
	cudaFree(weights);
	delete[] host_connection_points;
	delete[] host_weights;
	cudaDeviceSynchronize();

  cudaMalloc(&connection_neuron_i, sizeof(size_t) * connection_count);
	cudaMalloc(&weights, sizeof(field_t) * connection_count);
	cudaMalloc(&connection_points, sizeof(size_t) * connection_count);
	cudaDeviceSynchronize();

  cudaMemcpy(connection_neuron_i, vector_connection_neuron_i.data(), sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaMemcpy(weights, vector_weights.data(), sizeof(field_t) * connection_count, cudaMemcpyHostToDevice);
	cudaMemcpy(connection_points, vector_connection_points.data(), sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

void NeatConnections::remove_neuron(size_t neuron_i)
{
  size_t* tmp_connections_neuron_i = 0;
	size_t* tmp_connection_points = 0;
	field_t* tmp_weights = 0;
	field_t* tmp_biases = 0;

	size_t connection_count_until_deletion = 0;
	for (size_t i = 0; i < neuron_i; i++)
		connection_count_until_deletion += get_connection_count_at(i);

	size_t connection_count_after_deletion = 0;
	for (size_t i = neuron_i + 1; i < neuron_count; i++)
		connection_count_after_deletion += get_connection_count_at(i);

	size_t to_delete_connection_count = get_connection_count_at(neuron_i);
	
  cudaMalloc(&tmp_connections_neuron_i, sizeof(size_t) * (connection_count - to_delete_connection_count));
	cudaMalloc(&tmp_connection_points, sizeof(size_t) * (connection_count - to_delete_connection_count));
	cudaMalloc(&tmp_weights, sizeof(field_t) * (connection_count - to_delete_connection_count));
	cudaMalloc(&tmp_biases, sizeof(field_t) * (neuron_count - 1));
	cudaDeviceSynchronize();

  cudaMemcpy(tmp_connections_neuron_i, connection_neuron_i, sizeof(size_t) * (connection_count_until_deletion), cudaMemcpyDeviceToDevice);
  cudaMemcpy(tmp_connections_neuron_i + connection_count_until_deletion, connection_neuron_i + connection_count_until_deletion + to_delete_connection_count, sizeof(size_t) * connection_count_after_deletion, cudaMemcpyDeviceToDevice);
  
	cudaMemcpy(tmp_connection_points, connection_points, sizeof(size_t) * connection_count_until_deletion, cudaMemcpyDeviceToDevice);
	cudaMemcpy(tmp_connection_points + connection_count_until_deletion, connection_points + connection_count_until_deletion + to_delete_connection_count, sizeof(size_t) * connection_count_after_deletion, cudaMemcpyDeviceToDevice);

	cudaMemcpy(tmp_weights, weights, sizeof(field_t) * connection_count_until_deletion, cudaMemcpyDeviceToDevice);
	cudaMemcpy(tmp_weights + connection_count_until_deletion, weights + connection_count_until_deletion + to_delete_connection_count, sizeof(field_t) * connection_count_after_deletion, cudaMemcpyDeviceToDevice);

	cudaMemcpy(tmp_biases, biases, sizeof(field_t) * neuron_i, cudaMemcpyDeviceToDevice);
	cudaMemcpy(tmp_biases + neuron_i, biases + neuron_i + 1, sizeof(field_t) * (neuron_count - neuron_i - 1), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	cudaFree(connection_points);
	cudaFree(weights);
	cudaFree(biases);
  cudaFree(connection_neuron_i);
	cudaDeviceSynchronize();

  connection_neuron_i = tmp_connections_neuron_i;
	connection_points = tmp_connection_points;
	weights = tmp_weights;
	biases = tmp_biases;
	connection_count -= to_delete_connection_count;
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

