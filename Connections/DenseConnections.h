#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "IConnections.h"

class DenseConnections : public IConnections
{
public:
	size_t previous_layer_activations_start = 0;
	size_t previous_layer_length = 0;

	DenseConnections(size_t previous_layer_activations_start, size_t previous_layer_length, size_t neuron_count);
	DenseConnections();

	void linear_function(size_t activations_start, data_t* activations,
		data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
	);

	void calculate_derivative(
		size_t activations_start, data_t* activations,
		size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
	) override;

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		data_t* costs, size_t costs_start
	) override;

	void subtract_gradients(
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		data_t learning_rate, data_t gradient_clip
	) override;
	
	IConnections* connections_specific_clone() override;

	void specific_save(FILE* file) override;
	void load(FILE* file) override;

	size_t get_connection_count_at(size_t neuron_i) override;
};

