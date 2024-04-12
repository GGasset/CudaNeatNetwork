#include "ILayer.h"

#pragma once
class LSMTLayer : public ILayer
{
public:
	field_t* derivatives_until_memory_deletion = 0;
	size_t trained_steps_since_memory_deletion = 0;

	field_t* neuron_weights = 0;
	data_t* state = 0;

	void execute(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start
	) override;

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start, short calculate_derivatives,
		data_t* gradients, size_t next_gradients_start, size_t gradients_start,
		data_t* costs, size_t costs_start
	) override;
};

