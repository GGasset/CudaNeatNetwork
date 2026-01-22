
#pragma once
#include <cstddef>

struct nn_lens 
{
	// Per time_step
	size_t execution_values;
	// Per time_step
	size_t neurons;
	// Per time_step
	size_t gradients;
	// Per time_step
	size_t derivative;

	size_t layer_count;
};

struct layer_properties
{
	size_t per_neuron_hidden_state_count;

	size_t activations_start;
	size_t execution_values_start;
	size_t execution_values_per_neuron;

	size_t layer_derivative_count;
	size_t derivatives_per_neuron;
	size_t derivatives_start;

	size_t layer_gradient_count;
	size_t gradients_per_neuron;
	size_t gradients_start;

	size_t *per_neuron_gradients_start;
	size_t *per_connection_gradient_count;
};
