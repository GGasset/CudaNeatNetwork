
#include "Adam.h"

__device__ AdamOptimizer::AdamOptimizer()
{
	optimizer = Adam;
	values_per_parameter = 7;
}

__device__ void AdamOptimizer::initialize_optimizer_values(field_t* values)
{
	field_t tmp[7]{};
	tmp[0] = 0.9;
	tmp[1] = 0.999;
	tmp[2] = 1E-8;
	tmp[3] = 0;
	tmp[4] = 0;
	tmp[5] = tmp[0];
	tmp[6] = tmp[1];
	for (size_t i = 0; i < parameter_count; i++)
		memcpy(optimizer_values + values_per_parameter * i, tmp, sizeof(field_t) * values_per_parameter);
}

__device__ void AdamOptimizer::subtract_gradient(field_t* parameter, data_t gradient, size_t layer_parameter_i, gradient_hyperparameters hyperparameters)
{
	gradient /= hyperparameters.learning_rate;

	size_t values_starting_i = values_per_parameter * layer_parameter_i;

	data_t m = optimizer_values[values_starting_i + 3] =
		optimizer_values[values_starting_i] * optimizer_values[values_starting_i + 3] +
		(1 - optimizer_values[values_starting_i]) * gradient;

	data_t v = optimizer_values[values_starting_i + 4] =
		optimizer_values[values_starting_i + 1] * optimizer_values[values_starting_i + 4] +
		(1 - optimizer_values[values_starting_i + 1]) * gradient * gradient;


	optimizer_values[values_starting_i + 5] *= optimizer_values[values_starting_i];
	data_t bias_corrected_m = m / (1 - optimizer_values[values_starting_i + 5]);

	optimizer_values[values_starting_i + 6] *= optimizer_values[values_starting_i + 1];
	data_t bias_corrected_v = v / (1 - optimizer_values[values_starting_i + 6]);

	data_t to_add = -(hyperparameters.learning_rate * bias_corrected_m / (sqrt(abs(bias_corrected_v)) + optimizer_values[values_starting_i + 2]));

	atomicAdd(parameter, to_add);
}
