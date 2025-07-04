#include "IConnections.h"

void IConnections::mutate_fields(evolution_metadata evolution_values)
{
	float* arr0 = 0;
	cudaMalloc(&arr0, sizeof(float) * neuron_count * 3);
	cudaDeviceSynchronize();
	generate_random_values(arr0, neuron_count * 3, 0, 1);
	cudaDeviceSynchronize();

	mutate_field_array kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		biases, neuron_count,
		evolution_values.field_max_evolution, evolution_values.field_mutation_chance,
		arr0
	);
	cudaFree(arr0);
	cudaDeviceSynchronize();

	cudaMalloc(&arr0, sizeof(float) * connection_count * 3);
	cudaDeviceSynchronize();
	generate_random_values(arr0, connection_count * 3, 0, 1);
	cudaDeviceSynchronize();

	mutate_field_array kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
		weights, connection_count,
		evolution_values.field_max_evolution, evolution_values.field_mutation_chance,
		arr0
	);

	cudaFree(arr0);
	cudaDeviceSynchronize();
}

void IConnections::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
}

void IConnections::remove_neuron(size_t neuron_i)
{
}

void IConnections::adjust_to_removed_neuron(size_t neuron_i, std::vector<size_t>* removed_connections_neuron_i)
{
}

void IConnections::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability, std::vector<size_t>* added_connections_neuron_i)
{
}

void IConnections::IConnections_clone(IConnections* base)
{
	cudaMalloc(&base->weights, sizeof(field_t) * connection_count);
	cudaMalloc(&base->biases, sizeof(field_t) * neuron_count);
	cudaDeviceSynchronize();
	cudaMemcpy(base->weights, weights, sizeof(field_t) * connection_count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(base->biases, biases, sizeof(field_t) * neuron_count, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	base->neuron_count = neuron_count;
	base->connection_count = connection_count;
	base->contains_irregular_connections = contains_irregular_connections;
}

void IConnections::save(FILE* file)
{
	fwrite(&neuron_count, sizeof(size_t), 1, file);
	fwrite(&connection_count, sizeof(size_t), 1, file);
	fwrite(&contains_irregular_connections, sizeof(unsigned char), 1, file);

	specific_save(file);

	save_array(weights, connection_count, file, true);
	save_array(biases, neuron_count, file, true);
}


void IConnections::load_neuron_metadata(FILE* file)
{
	fread(&neuron_count, sizeof(size_t), 1, file);
	fread(&connection_count, sizeof(size_t), 1, file);
	fread(&contains_irregular_connections, sizeof(unsigned char), 1, file);
}

void IConnections::load_IConnections_data(FILE* file)
{
	weights = load_array<field_t>(connection_count, file, true);
	biases = load_array<field_t>(neuron_count, file, true);
}

void IConnections::deallocate()
{
	cudaFree(weights);
	cudaFree(biases);
	specific_deallocate();
	cudaDeviceSynchronize();
}

void IConnections::specific_deallocate()
{
}
