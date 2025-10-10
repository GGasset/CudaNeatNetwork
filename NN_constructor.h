#pragma once

#include "NN.h"
#include <vector>

class NN_constructor
{
private:
	size_t layer_count = 0;
	std::vector<NeuronTypes> neuron_types;
	std::vector<ConnectionTypes> connection_types;
	std::vector<size_t> layer_lengths;
	std::vector<ActivationFunctions> activations;
public:
	NN_constructor();

	NN_constructor append_layer(ConnectionTypes connections_type, NeuronTypes neurons_type, size_t neuron_count, ActivationFunctions activation = ActivationFunctions::sigmoid);
	NN* construct(size_t input_length, optimizer_hyperparameters optimizer_options, bool stateful = false);
};
