#include "ILayer.h"

size_t ILayer::get_neuron_count()
{
	return neuron_count;
}

void ILayer::set_neuron_count(size_t neuron_count)
{
	this->neuron_count = neuron_count;
	connections->neuron_count = neuron_count;
}

void ILayer::initialize_fields(size_t connection_count, size_t neuron_count, bool initialize_connection_associated_gradient_count)
{
	size_t* neuron_gradients_starts = new size_t[neuron_count];
	size_t* connection_associated_gradient_counts = 0;
	if (initialize_connection_associated_gradient_count)
		connection_associated_gradient_counts = new size_t[neuron_count];
	size_t gradient_count = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t neuron_connection_count = connections->get_connection_count_at(i);

		if (initialize_connection_associated_gradient_count)
			connection_associated_gradient_counts[i] = neuron_connection_count + 1;
		neuron_gradients_starts[i] = gradient_count;
		
		gradient_count += neuron_connection_count + 1 + gradients_per_neuron;
	}

	cudaMalloc(&this->neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaMemcpy(this->neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	delete[] neuron_gradients_starts;

	if (initialize_connection_associated_gradient_count)
	{
		cudaMalloc(&this->connection_associated_gradient_counts, sizeof(size_t) * neuron_count);
		cudaMemcpy(this->connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
		delete[] connection_associated_gradient_counts;
	}

	layer_specific_initialize_fields(connection_count, neuron_count);
	cudaDeviceSynchronize();
}

void ILayer::layer_specific_initialize_fields(size_t connection_count, size_t neuron_count)
{
}

void ILayer::ILayerClone(ILayer* base_layer)
{
	IConnections* cloned_connections = connections->connections_specific_clone();
	connections->IConnections_clone(cloned_connections);
	base_layer->connections = cloned_connections;

	base_layer->set_neuron_count(get_neuron_count());

	base_layer->execution_values_per_neuron = execution_values_per_neuron;
	
	base_layer->layer_derivative_count = layer_derivative_count;
	base_layer->derivatives_per_neuron = derivatives_per_neuron;

	base_layer->layer_gradient_count = layer_gradient_count;
	
	cudaMalloc(&base_layer->neuron_gradients_starts, sizeof(size_t) * get_neuron_count());
	if (connection_associated_gradient_counts)
		cudaMalloc(&base_layer->connection_associated_gradient_counts, sizeof(size_t) * get_neuron_count());
	cudaDeviceSynchronize();

	cudaMemcpy(base_layer->neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * get_neuron_count(), cudaMemcpyDeviceToDevice);
	if (connection_associated_gradient_counts)
		cudaMemcpy(base_layer->connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
}

void ILayer::save(FILE* file)
{
	fwrite(&neuron_count, sizeof(size_t), 1, file);
	fwrite(&execution_values_per_neuron, sizeof(size_t), 1, file);
	fwrite(&layer_derivative_count, sizeof(size_t), 1, file);
	fwrite(&derivatives_per_neuron, sizeof(size_t), 1, file);
	fwrite(&layer_gradient_count, sizeof(size_t), 1, file);
	
	size_t *host_neuron_gradients_starts, *host_connection_gradient_counts;
	host_neuron_gradients_starts = new size_t[neuron_count];
	host_connection_gradient_counts = new size_t[neuron_count];

	cudaMemcpy(host_neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_connection_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	fwrite(host_neuron_gradients_starts, sizeof(size_t), neuron_count, file);
	fwrite(host_connection_gradient_counts, sizeof(size_t), neuron_count, file);
	delete[] host_neuron_gradients_starts;
	delete[] host_connection_gradient_counts;

	specific_save(file);
}

void ILayer::ILayer_load(FILE* file)
{
	fread(&neuron_count, sizeof(size_t), 1, file);
	fread(&execution_values_per_neuron, sizeof(size_t), 1, file);
	fread(&layer_derivative_count, sizeof(size_t), 1, file);
	fread(&derivatives_per_neuron, sizeof(size_t), 1, file);
	fread(&layer_gradient_count, sizeof(size_t), 1, file);

	size_t* host_neuron_gradients_starts = new size_t[neuron_count];
	size_t* host_connection_associated_gradient_counts = new size_t[neuron_count];

	fread(host_neuron_gradients_starts, sizeof(size_t), neuron_count, file);
	fread(host_connection_associated_gradient_counts, sizeof(size_t), neuron_count, file);

	cudaMalloc(&neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaMalloc(&connection_associated_gradient_counts, sizeof(size_t) * neuron_count);
	cudaDeviceSynchronize();

	cudaMemcpy(neuron_gradients_starts, host_neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(connection_associated_gradient_counts, host_connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	delete[] host_neuron_gradients_starts;
	delete[] host_connection_associated_gradient_counts;
}

void ILayer::deallocate()
{
	connections->deallocate();
	layer_specific_deallocate();
	cudaDeviceSynchronize();
	delete connections;
}

void ILayer::layer_specific_deallocate()
{

}

void ILayer::mutate_fields(evolution_metadata evolution_values)
{
}

void ILayer::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
	size_t added_connection_count = connections->connection_count;
	connections->add_neuron(previous_layer_length, previous_layer_activations_start, previous_layer_connection_probability, min_connections);
	added_connection_count = connections->connection_count - added_connection_count;
	set_neuron_count(neuron_count + 1);

	if (connection_associated_gradient_counts)
	{
		size_t* tmp_connection_associated_gradient_counts = new size_t[neuron_count];
		cudaMemcpy(tmp_connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count - 1, cudaMemcpyDeviceToHost);
		tmp_connection_associated_gradient_counts[neuron_count - 1] = added_connection_count;
		cudaFree(connection_associated_gradient_counts);
		cudaMalloc(&connection_associated_gradient_counts, sizeof(size_t) * neuron_count);
		cudaMemcpy(connection_associated_gradient_counts, tmp_connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
		delete[] tmp_connection_associated_gradient_counts;
	}

	if (neuron_gradients_starts)
	{
		size_t* tmp_neuron_gradients_starts = new size_t[neuron_count];
		cudaMemcpy(tmp_neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * neuron_count - 1, cudaMemcpyDeviceToHost);
		tmp_neuron_gradients_starts[neuron_count - 1] = tmp_neuron_gradients_starts[neuron_count - 2] + added_connection_count + gradients_per_neuron;
		cudaFree(neuron_gradients_starts);
		cudaMalloc(&neuron_gradients_starts, sizeof(size_t) * neuron_count);
		cudaMemcpy(neuron_gradients_starts, tmp_neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
		delete[] tmp_neuron_gradients_starts;
	}

	layer_derivative_count += derivatives_per_neuron;
	layer_gradient_count += added_connection_count + gradients_per_neuron + 1;

	layer_specific_add_neuron();
}

void ILayer::layer_specific_add_neuron()
{

}

void ILayer::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability)
{
	auto added_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_added_neuron(added_neuron_i, connection_probability, &added_connections_neuron_i);
	for (size_t i = 0; i < added_connections_neuron_i.size(); i++)
	{
		layer_gradient_count++;
		size_t added_connection_neuron_i = added_connections_neuron_i[i];
		size_t remaining_neuron_count = neuron_count - added_connection_neuron_i - 1;
		if (remaining_neuron_count)
		{
			if (connection_associated_gradient_counts)
				add_to_array kernel(1, 1) (
					connection_associated_gradient_counts + added_connection_neuron_i, 1, 1
				);
			if (neuron_gradients_starts)
				add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
					neuron_gradients_starts + added_connection_neuron_i + 1, remaining_neuron_count, 1
				);
		}
	}
}

void ILayer::remove_neuron(size_t layer_neuron_i)
{
	size_t removed_connection_count = connections->connection_count;
	connections->remove_neuron(layer_neuron_i);
	removed_connection_count -= connections->connection_count;

	set_neuron_count(neuron_count - 1);
	size_t removed_gradients = removed_connection_count + gradients_per_neuron + 1;
	layer_gradient_count -= removed_gradients;
	layer_derivative_count -= derivatives_per_neuron;

	if (neuron_gradients_starts)
	{
		neuron_gradients_starts = 
			cuda_remove_elements(neuron_gradients_starts, neuron_count + 1, layer_neuron_i, 1, true);
		add_to_array kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (neuron_gradients_starts, neuron_count, removed_gradients);
	}

	if (connection_associated_gradient_counts)
		connection_associated_gradient_counts =
			cuda_remove_elements(connection_associated_gradient_counts, neuron_count + 1, layer_neuron_i, 1, true);
	cudaDeviceSynchronize();
	layer_specific_remove_neuron(layer_neuron_i);
}

void ILayer::layer_specific_remove_neuron(size_t layer_neuron_i)
{
}

void ILayer::adjust_to_removed_neuron(size_t neuron_i)
{
	auto removed_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_removed_neuron(neuron_i, &removed_connections_neuron_i);
	for (size_t i = 0; i < removed_connections_neuron_i.size(); i++)
	{
		layer_gradient_count--;
		size_t removed_connection_neuron_i = removed_connections_neuron_i[i];
		size_t remaining_neuron_count = neuron_count - removed_connection_neuron_i - 1;
		if (remaining_neuron_count)
		{
			if (connection_associated_gradient_counts)
				add_to_array kernel(1, 1) (
					connection_associated_gradient_counts + removed_connection_neuron_i, 1, -1
				);
			if (neuron_gradients_starts)
				add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
					neuron_gradients_starts + removed_connection_neuron_i + 1, remaining_neuron_count, -1
				);
		}
	}
}

void ILayer::delete_memory()
{
}
