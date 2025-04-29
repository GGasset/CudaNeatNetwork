
#include "Optimizers.h"

void Optimizer_device::alloc_optimizer_values(size_t param_count, bool copy_old_values)
{
	size_t old_param_count = parameter_count;
	field_t* old_optimizer_values = optimizer_values;
	parameter_count = param_count;

	cudaMalloc(&optimizer_values, sizeof(field_t) * param_count * values_per_parameter);
	initialize_optimizer_values(optimizer_values);
	if (copy_old_values && old_optimizer_values && optimizer_values)
		cudaMemcpy(optimizer_values, old_optimizer_values, sizeof(field_t) * h_min(param_count, old_param_count) * values_per_parameter, cudaMemcpyDeviceToDevice);
	cudaFree(old_optimizer_values);
}

void Optimizer_device::initialize_optimizer_values(field_t* values)
{
	cudaMemset(values, 0, sizeof(field_t) * parameter_count * values_per_parameter);
}

void Optimizer_device::hyperparameter_subtract_gradient(field_t* parameter, data_t gradient, size_t layer_parameter_i, gradient_hyperparameters hyperparameters)
{
	gradient *= hyperparameters.learning_rate;
	hyperparameters.gradient_clip = abs(hyperparameters.gradient_clip) * (-1 + 2 * (gradient >= 0));
	gradient = device_closest_to_zero(gradient, hyperparameters.gradient_clip);
	subtract_gradient(parameter, gradient, layer_parameter_i);
}

void Optimizer_device::subtract_gradient(field_t* parameter, data_t gradient, size_t layer_parameter_i)
{
	*parameter -= gradient;
}
