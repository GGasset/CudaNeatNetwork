
#include "Adam.h"

AdamOptimizer::AdamOptimizer()
{
	values_per_parameter = 5;
}

void AdamOptimizer::initialize_optimizer_values(field_t* values)
{
	field_t tmp[5]{};
	tmp[0] = 0.9;
	tmp[1] = 0.999;
	tmp[2] = 10E-8;
	tmp[3] = 0;
	tmp[4] = 0;
	for (size_t i = 0; i < values_per_parameter; i++)
		cudaMemcpy(optimizer_values + values_per_parameter * i, tmp, sizeof(field_t) * values_per_parameter, cudaMemcpyDeviceToDevice);
}

void AdamOptimizer::subtract_gradient(field_t* parameter, data_t gradient, size_t layer_parameter_i, gradient_hyperparameters hyperparameters)
{
	gradient /= hyperparameters.learning_rate;

	size_t values_starting_i = values_per_parameter * layer_parameter_i;

	data_t m = optimizer_values[values_starting_i + 3] =
		optimizer_values[values_starting_i] * optimizer_values[values_starting_i + 3] +
		(1 - optimizer_values[values_starting_i]) * gradient;

	data_t v = optimizer_values[values_starting_i + 4] =
		optimizer_values[values_starting_i + 1] * optimizer_values[values_starting_i + 4] +
		(1 - optimizer_values[values_starting_i + 1]) * gradient * gradient;


	optimizer_values[values_starting_i] *= optimizer_values[values_starting_i];
	data_t bias_corrected_m = m / (1 - optimizer_values[values_starting_i]);

	optimizer_values[values_starting_i] *= optimizer_values[values_starting_i];
	data_t bias_corrected_v = v / (1 - optimizer_values[values_starting_i]);

	*parameter -= hyperparameters.learning_rate * bias_corrected_m / (sqrt(bias_corrected_v) + optimizer_values[values_starting_i + 2]);
}
