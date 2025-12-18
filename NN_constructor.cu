#include "NN_constructor.h"

NN_constructor::NN_constructor()
{

}

NN_constructor NN_constructor::append_layer(
	ConnectionTypes connections_type, NeuronTypes neurons_type, size_t neuron_count, ActivationFunctions activation,
	initialization_parameters weight_init, initialization_parameters bias_init, initialization_parameters layer_weight_init
)
{
	connection_types.push_back(connections_type);
	neuron_types.push_back(neurons_type);
	layer_lengths.push_back(neuron_count);
	activations.push_back(activation);
	initialization.push_back({weight_init, bias_init, layer_weight_init});
	layer_count++;
	return *this;
}

NN* NN_constructor::construct(
	size_t input_length, optimizer_hyperparameters optimizer_options, bool stateful,
	initialization_parameters weight,
	initialization_parameters bias,
	initialization_parameters layer_weight
)
{
	ILayer** layers = new ILayer*[layer_count];
	size_t previous_layer_activations_start = 0;
	for (int i = 0; i < layer_count; i++)
	{
		size_t previous_layer_length = i ? layer_lengths[i - 1] : input_length;
		size_t layer_length = layer_lengths[i];
		ActivationFunctions activation = activations[i];
		auto [weight_init, bias_init, layer_weight_init] = initialization[i];

		IConnections* connections = 0;
		ILayer* layer = 0;
		switch (connection_types[i])
		{
			case ConnectionTypes::Dense:
				connections = new DenseConnections(previous_layer_activations_start, previous_layer_length, layer_length, weight_init, bias_init);
				break;
			case ConnectionTypes::NEAT:
				connections = new NeatConnections(previous_layer_activations_start, previous_layer_length, layer_length, weight_init, bias_init);
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
				layer = new LSTMLayer(connections, layer_length, layer_weight_init);
				break;
			default:
				break;
		}
		
		Optimizers optimizer = Optimizers(layer->get_weight_count(), optimizer_options);
		layer->optimizer = optimizer;
		
		layers[i] = layer;
		previous_layer_activations_start += i ? layer_length : input_length;
	}
	NN* n = new NN(layers, input_length, layer_count, weight, bias, layer_weight);
	n->stateful = stateful;
	n->optimizer_initialization = optimizer_options;
	return n;
}
