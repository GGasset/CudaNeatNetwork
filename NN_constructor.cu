#include "NN_constructor.h"

NN_constructor::NN_constructor()
{

}

NN_constructor NN_constructor::append_layer(ConnectionTypes connections_type, NeuronTypes neurons_type, size_t neuron_count, ActivationFunctions activation)
{
	connection_types.push_back(connections_type);
	neuron_types.push_back(neurons_type);
	layer_lengths.push_back(neuron_count);
	activations.push_back(activation);
	layer_count++;
	return *this;
}

NN* NN_constructor::construct(size_t input_length, optimizers_enum e_optimizer, bool stateful)
{
	ILayer** layers = new ILayer*[layer_count];
	size_t previous_layer_activations_start = 0;
	for (int i = 0; i < layer_count; i++)
	{
		size_t previous_layer_length = i ? layer_lengths[i - 1] : input_length;
		size_t layer_length = layer_lengths[i];
		ActivationFunctions activation = activations[i];
		IConnections* connections = 0;
		ILayer* layer = 0;
		switch (connection_types[i])
		{
			case ConnectionTypes::Dense:
				connections = new DenseConnections(previous_layer_activations_start, previous_layer_length, layer_length);
				break;
			case ConnectionTypes::NEAT:
				connections = new NeatConnections(previous_layer_activations_start, previous_layer_length, layer_length);
				break;
			default:
				break;
		}
		switch (neuron_types[i])
		{
			case NeuronTypes::Neuron:
				layer = new NeuronLayer(connections, layer_length, activation);
				break;
			case NeuronTypes::LSTM:
				layer = new LSTMLayer(connections, layer_length);
				break;
			default:
				break;
		}
		
		IOptimizer* optimizer = host_optimizer_init(e_optimizer, layer->get_weight_count());
		layer->optimizer = optimizer;
		
		layers[i] = layer;
		previous_layer_activations_start += i ? layer_length : input_length;
	}
	NN* n = new NN(layers, input_length, layer_count);
	n->stateful = stateful;
	return n;
}
