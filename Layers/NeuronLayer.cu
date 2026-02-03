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
	properties.execution_values_per_neuron = 1 + (activation == softmax);
	properties.layer_gradient_count = connections->connection_count + neuron_count;

	initialize_fields(connections->connection_count, neuron_count, false);
}

NeuronLayer::NeuronLayer()
{
	layer_type = NeuronTypes::Neuron;
}

void NeuronLayer::layer_specific_deallocate()
{
	cudaFree(properties.per_neuron_gradients_start);
	if (properties.per_connection_gradient_count)
		cudaFree(properties.per_connection_gradient_count);
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

void NeuronLayer::execute(size_t t_count, data_t *activations, data_t *execution_values, nn_lens lens, size_t timestep_gap)
{
	connections->linear_function(t_count, activations, execution_values, properties, lens, timestep_gap);


}

void NeuronLayer::execute(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start
)
{
	connections->linear_function(activations_start, activations,
		execution_values, execution_values_start, properties.execution_values_start, properties.execution_values_per_neuron
	);
	activation_function (
		activation,
		activations, activations_start, properties.activations_start, true,
		execution_values, execution_values_start, properties.execution_values_start, properties.execution_values_per_neuron, 0, 0, 0,
		neuron_count
	);
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
	neuron_gradient_calculation(
		execution_values, execution_values_start, properties.execution_values_start, properties.execution_values_per_neuron,
		gradients, gradients_start, properties.gradients_start, properties.per_neuron_gradients_start,
		costs, costs_start, properties.activations_start,
		activation,
		neuron_count
	);
	connections->calculate_gradients(
		activations, activations_start, gradients, gradients_start, properties.gradients_start, properties.per_neuron_gradients_start,
		costs, costs_start
	);
	cudaDeviceSynchronize();
}

void NeuronLayer::subtract_gradients(data_t* gradients, size_t gradients_start, gradient_hyperparameters hyperparameters)
{
	connections->subtract_gradients(
		gradients, gradients_start, properties.gradients_start, properties.per_neuron_gradients_start,
		hyperparameters, optimizer
	);
}

#endif
