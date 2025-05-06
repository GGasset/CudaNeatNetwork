#ifndef NEURONLAYER_DEFINITIONS
#define NEURONLAYER_DEFINITIONS

#include "NeuronLayer.h"
#include <stdio.h>

NeuronLayer::NeuronLayer(IConnections* connections, size_t neuron_count, ActivationFunctions activation)
{
	layer_type = NeuronTypes::Neuron;

	this->connections = connections;
	set_neuron_count(neuron_count);
	this->activation = activation;
	execution_values_per_neuron = 1;
	layer_gradient_count = connections->connection_count + neuron_count;

	initialize_fields(connections->connection_count, neuron_count, false);
}

NeuronLayer::NeuronLayer()
{
	layer_type = NeuronTypes::Neuron;
}

void NeuronLayer::layer_specific_deallocate()
{
	cudaFree(neuron_gradients_starts);
	if (connection_associated_gradient_counts)
		cudaFree(connection_associated_gradient_counts);
}

ILayer* NeuronLayer::layer_specific_clone()
{
	NeuronLayer* layer = new NeuronLayer();
	layer->activation = activation;
	return layer;
}

void NeuronLayer::specific_save(FILE* file)
{
	size_t activation_function = (size_t)activation;
	fwrite(&activation_function, sizeof(size_t), 1, file);
}

void NeuronLayer::load(FILE* file)
{
	ILayer_load(file);
	
	size_t activation_function = 0;
	fread(&activation_function, sizeof(size_t), 1, file);
	activation = (ActivationFunctions)activation_function;
}

void NeuronLayer::execute(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start
)
{
	connections->linear_function(activations_start, activations,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron
	);
	switch (activation)
	{
	case ActivationFunctions::sigmoid:
		sigmoid_activation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
			activations, activations_start, layer_activations_start, true,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron, 0, 0, 0,
			neuron_count
		);
		break;
	case ActivationFunctions::_tanh:
		tanh_activation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
			activations, activations_start, layer_activations_start, true,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron, 0, 0, 0,
			neuron_count
		);
		break;
	default:
		break;
	}
	cudaDeviceSynchronize();
}

void NeuronLayer::calculate_gradients(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start,
	data_t* derivatives, size_t derivatives_start,
	data_t* gradients, size_t next_gradients_start, size_t gradients_start,
	data_t* costs, size_t costs_start
)
{
	neuron_gradient_calculation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		execution_values, execution_values_start, execution_values_layer_start,
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start, layer_activations_start,
		activation,
		neuron_count
	);
	cudaDeviceSynchronize();
	connections->calculate_gradients(
		activations, activations_start, gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start
	);
	cudaDeviceSynchronize();
}

void NeuronLayer::subtract_gradients(data_t* gradients, size_t gradients_start, data_t learning_rate, data_t gradient_clip)
{
	connections->subtract_gradients(
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		learning_rate, gradient_clip
	);
}

#endif
