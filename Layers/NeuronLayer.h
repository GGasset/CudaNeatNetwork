#pragma once

#include "ILayer.h"

class NeuronLayer : public ILayer
{
protected:
	ActivationFunctions activation = ActivationFunctions::sigmoid;

public:
	NeuronLayer(IConnections* connections, size_t neuron_count, ActivationFunctions activation);
	NeuronLayer();
	
	void layer_specific_deallocate() override;

	ILayer* layer_specific_clone() override;

	void specific_save(FILE* file) override;
	void load(FILE* file) override;

	void execute(
		size_t t_count, data_t *activations, data_t *execution_values,
		nn_lens lens, size_t timestep_gap
	) override;

	void backpropagate(
		size_t t_count, data_t *activations, data_t *execution_values, data_t *gradients, data_t *costs, data_t *derivatives,
		nn_lens lens, size_t timestep_gap_len
	) override;

	void execute(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start
	) override;

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start,
		data_t* derivatives, size_t derivatives_start,
		data_t* gradients, size_t next_gradients_start, size_t gradients_start,
		data_t* costs, size_t costs_start
	) override;

	void subtract_gradients(
		data_t* gradients, size_t gradients_start, gradient_hyperparameters hyperparameters
	) override;

	void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
		data_t* execution_values, size_t execution_values_start
	) override
	{

	}
};

