#include "ILayer.h"

size_t ILayer::get_neuron_count()
{
	return neuron_count;
}

void ILayer::set_neuron_count(size_t neuron_count)
{
	this->neuron_count = neuron_count;
	connections->neuron_count = neuron_count;
	properties.neuron_count = neuron_count;
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
		
		gradient_count += neuron_connection_count + 1 + properties.gradients_per_neuron;
	}

	cudaMalloc(&this->properties.per_neuron_gradients_start, sizeof(size_t) * neuron_count);
	cudaMemcpy(this->properties.per_neuron_gradients_start, neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	delete[] neuron_gradients_starts;

	if (initialize_connection_associated_gradient_count)
	{
		cudaMalloc(&this->properties.per_connection_gradient_count, sizeof(size_t) * neuron_count);
		cudaMemcpy(this->properties.per_connection_gradient_count, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
		delete[] connection_associated_gradient_counts;
	}

	layer_specific_initialize_fields(connection_count, neuron_count);
	cudaDeviceSynchronize();

	set_neuron_count(neuron_count);
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

	base_layer->properties.execution_values_per_neuron = properties.execution_values_per_neuron;
	
	base_layer->properties.layer_derivative_count = properties.layer_derivative_count;
	base_layer->properties.derivatives_per_neuron = properties.derivatives_per_neuron;

	base_layer->properties.layer_gradient_count = properties.layer_gradient_count;

	base_layer->properties.per_neuron_hidden_state_count = properties.per_neuron_hidden_state_count;

	base_layer->optimizer = optimizer.Clone();
	
	cudaMalloc(&base_layer->properties.per_neuron_gradients_start, sizeof(size_t) * get_neuron_count());
	if (properties.per_connection_gradient_count)
		cudaMalloc(&base_layer->properties.per_connection_gradient_count, sizeof(size_t) * get_neuron_count());
	cudaDeviceSynchronize();

	cudaMemcpy(base_layer->properties.per_neuron_gradients_start, properties.per_neuron_gradients_start, sizeof(size_t) * get_neuron_count(), cudaMemcpyDeviceToDevice);
	if (properties.per_connection_gradient_count)
		cudaMemcpy(base_layer->properties.per_connection_gradient_count, properties.per_connection_gradient_count, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
}

void ILayer::save(FILE* file)
{
	fwrite(&neuron_count, sizeof(size_t), 1, file);
	fwrite(&properties.execution_values_per_neuron, sizeof(size_t), 1, file);
	fwrite(&properties.layer_derivative_count, sizeof(size_t), 1, file);
	fwrite(&properties.derivatives_per_neuron, sizeof(size_t), 1, file);
	fwrite(&properties.layer_gradient_count, sizeof(size_t), 1, file);

	bool contains_connection_gradient_counts = properties.per_connection_gradient_count != 0;
	fwrite(&contains_connection_gradient_counts, sizeof(bool), 1, file);
	
	save_array(properties.per_neuron_gradients_start, neuron_count, file, true);

	if (contains_connection_gradient_counts)
		save_array(properties.per_connection_gradient_count, neuron_count, file, true);

	optimizer.save(file);

	specific_save(file);
}

void ILayer::ILayer_load(FILE* file)
{
	fread(&neuron_count, sizeof(size_t), 1, file);
	fread(&properties.execution_values_per_neuron, sizeof(size_t), 1, file);
	fread(&properties.layer_derivative_count, sizeof(size_t), 1, file);
	fread(&properties.derivatives_per_neuron, sizeof(size_t), 1, file);
	fread(&properties.layer_gradient_count, sizeof(size_t), 1, file);

	bool contains_connection_associated_gradient_counts = 0;
	fread(&contains_connection_associated_gradient_counts, sizeof(bool), 1, file);

	properties.per_neuron_gradients_start = load_array<size_t>(neuron_count, file, true);

	if (contains_connection_associated_gradient_counts)
		properties.per_connection_gradient_count = load_array<size_t>(neuron_count, file, true);

	optimizer = Optimizers::load(file);
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

void ILayer::backpropagate(
	size_t t_count, data_t *activations, data_t *execution_values, data_t *gradients, data_t *costs, 
	data_t *states, nn_lens lens, size_t timestep_gap
)
{
	backpropagate(t_count, activations, execution_values, gradients, costs, lens, timestep_gap);
}

void ILayer::calculate_derivatives(
	size_t t_count, data_t *activations, data_t *execution_values, data_t *derivatives, 
	nn_lens lens, size_t timestep_gap
)
{
}

void ILayer::mutate_fields(evolution_metadata evolution_values)
{
}

void ILayer::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
	size_t previous_connection_count = connections->connection_count;
	connections->add_neuron(previous_layer_length, previous_layer_activations_start, previous_layer_connection_probability, min_connections);
	size_t added_connection_count = connections->connection_count - previous_connection_count;

	if (properties.per_connection_gradient_count)
		properties.per_connection_gradient_count = cuda_push_back(properties.per_connection_gradient_count, sizeof(size_t) * neuron_count, 1 + added_connection_count, true);

	if (properties.per_neuron_gradients_start)
	{
		size_t* tmp_neuron_gradients_starts = new size_t[neuron_count + 1];
		cudaMemcpy(tmp_neuron_gradients_starts, properties.per_neuron_gradients_start, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToHost);
		tmp_neuron_gradients_starts[neuron_count] = tmp_neuron_gradients_starts[neuron_count - 1] + 1 + connections->get_connection_count_at(neuron_count - 1) + properties.gradients_per_neuron;

		cudaFree(properties.per_neuron_gradients_start);
		cudaMalloc(&properties.per_neuron_gradients_start, sizeof(size_t) * (neuron_count + 1));
		cudaMemcpy(properties.per_neuron_gradients_start, tmp_neuron_gradients_starts, sizeof(size_t) * (neuron_count + 1), cudaMemcpyHostToDevice);
		delete[] tmp_neuron_gradients_starts;
	}

	properties.layer_derivative_count += properties.derivatives_per_neuron;
	properties.layer_gradient_count += added_connection_count + properties.gradients_per_neuron + 1;

	size_t layer_added_parameter_count = get_weight_count();
	layer_specific_add_neuron();
	layer_added_parameter_count = get_weight_count() - layer_added_parameter_count;
	layer_added_parameter_count -= neuron_count + connections->connection_count;

	optimizer.add_parameters(1, neuron_count);
	optimizer.add_parameters(added_connection_count, neuron_count + previous_connection_count);
	optimizer.add_parameters(layer_added_parameter_count, -1);

	set_neuron_count(neuron_count + 1);
}

void ILayer::layer_specific_add_neuron()
{

}

void ILayer::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability)
{
	size_t previous_connection_count = connections->connection_count;
	auto added_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_added_neuron(added_neuron_i, connection_probability, &added_connections_neuron_i);
	optimizer.add_parameters(added_connections_neuron_i.size(), neuron_count + previous_connection_count);

	for (size_t i = 0; i < added_connections_neuron_i.size(); i++)
	{
		properties.layer_gradient_count++;
		size_t added_connection_neuron_i = added_connections_neuron_i[i];
		size_t remaining_neuron_count = neuron_count - added_connection_neuron_i - 1;
		if (remaining_neuron_count)
		{
			if (properties.per_connection_gradient_count)
				add_to_array kernel(1, 1) (
					properties.per_connection_gradient_count + added_connection_neuron_i, 1, 1
				);
			if (properties.per_neuron_gradients_start)
				add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
					properties.per_neuron_gradients_start + added_connection_neuron_i + 1, remaining_neuron_count, 1
				);
		}
	}
}

void ILayer::remove_neuron(size_t layer_neuron_i)
{
	std::vector<size_t> removed_connections_i;
	size_t removed_connection_count = connections->connection_count;
	connections->remove_neuron(layer_neuron_i, &removed_connections_i);
	removed_connection_count -= connections->connection_count;

	size_t removed_gradients = removed_connection_count + properties.gradients_per_neuron + 1;
	properties.layer_gradient_count -= removed_gradients;
	properties.layer_derivative_count -= properties.derivatives_per_neuron;

	if (properties.per_neuron_gradients_start)
	{
		properties.per_neuron_gradients_start = 
			cuda_remove_elements(properties.per_neuron_gradients_start, neuron_count, layer_neuron_i, 1, true);
		
		size_t after_deletion_neuron_count = get_neuron_count() - layer_neuron_i - 1;
		if (after_deletion_neuron_count)
			add_to_array kernel(after_deletion_neuron_count / 32 + (after_deletion_neuron_count % 32 > 0), 32) (
				properties.per_neuron_gradients_start + layer_neuron_i, after_deletion_neuron_count, -(int)removed_gradients
			);
	}

	if (properties.per_connection_gradient_count)
		properties.per_connection_gradient_count =
			cuda_remove_elements(properties.per_connection_gradient_count, neuron_count, layer_neuron_i, 1, true);
	cudaDeviceSynchronize();

	layer_specific_remove_neuron(layer_neuron_i);

	for (size_t i = removed_connections_i.size(); i > 0; i++)
		optimizer.remove_parameters(1, neuron_count + removed_connections_i[i - 1]);
	
	optimizer.remove_parameters(1, layer_neuron_i);


	set_neuron_count(neuron_count - 1);
}

void ILayer::layer_specific_remove_neuron(size_t layer_neuron_i)
{
}

void ILayer::adjust_to_removed_neuron(size_t neuron_i)
{
	size_t before_connection_count = connections->connection_count;
	auto removed_connections_neuron_i = std::vector<size_t>();
	auto removed_connections_i = std::vector<size_t>();
	connections->adjust_to_removed_neuron(neuron_i, &removed_connections_neuron_i, &removed_connections_i);
	for (size_t i = 0; i < removed_connections_neuron_i.size(); i++)
	{
		optimizer.remove_parameters(1, neuron_count + before_connection_count - removed_connections_i[i]);
		properties.layer_gradient_count--;
		size_t removed_connection_neuron_i = removed_connections_neuron_i[i];
		size_t remaining_neuron_count = neuron_count - removed_connection_neuron_i - 1;
		if (remaining_neuron_count)
		{
			if (properties.per_connection_gradient_count)
				add_to_array kernel(1, 1) (
					properties.per_connection_gradient_count + removed_connection_neuron_i, 1, -1
				);
			if (properties.per_neuron_gradients_start)
				add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
					properties.per_neuron_gradients_start + removed_connection_neuron_i + 1, remaining_neuron_count, -1
				);
		}
	}
}

void ILayer::delete_memory()
{
}
