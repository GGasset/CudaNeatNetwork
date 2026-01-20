
#pragma once
#include <cstddef>

struct nn_lens {

	// Per time_step
	size_t execution_value_count;
	// Per time_step
	size_t neuron_count;
	// Per time_step
	size_t gradient_count;
	// Per time_step
	size_t derivative_count;


};

