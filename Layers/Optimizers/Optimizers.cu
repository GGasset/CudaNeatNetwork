
#include "Optimizers.h"

void Optimizers::copy_values_of(size_t optimizer_i, Optimizers dst)
{
	Optimizer_values dst_values = dst.optimizer_values[optimizer_i];
	Optimizer_values src_values = optimizer_values[optimizer_i];

	if (dst_values.value_count_per_parameter != src_values.value_count_per_parameter) throw;
	if (dst.parameter_count != parameter_count) throw;
	if (!dst_values.values || !src_values.values) return ;

	size_t arr_len = src_values.value_count_per_parameter * parameter_count;
	cudaMemcpy(dst_values.values, src_values.values, sizeof(data_t) * arr_len, cudaMemcpyDeviceToDevice);
}

void Optimizers::save_values_of(size_t optimizer_i, FILE *file)
{
	Optimizer_values optimizer = optimizer_values[optimizer_i];
	size_t len = parameter_count * optimizer.value_count_per_parameter;
	if (!len) return;
	save_array(optimizer.values, len, file, true);
}

Optimizer_values Optimizers::load_values_of(size_t optimizer_i, FILE *file)
{
	Optimizer_values out;

	size_t values_per_param = optimizer_values_per_parameter[optimizer_i];
	out.value_count_per_parameter = values_per_param;
	size_t arr_len = values_per_param * parameter_count;
	if (!arr_len) return out;

	out.values = load_array<data_t>(arr_len, file, true);
}

void Optimizers::initialize_values(long parameter_start_i, size_t parameter_count)
{
	data_t *adam = optimizer_values[Adam].values;
	adam[0] = initialization_hyperparameters.adam.beta_1;
	adam[1] = initialization_hyperparameters.adam.beta_2;
	adam[2] = initialization_hyperparameters.adam.epsilon;
	adam[5] = adam[0];
	adam[6] = adam[1];
}

void Optimizers::allocate_values()
{
	for (size_t i = 0; i < optimizers_enum::last_optimizer_entry; i++)
	{
		if (!optimizer_values[i].value_count_per_parameter)
			continue;
		cudaFree(optimizer_values[i].values);

		size_t len = parameter_count * optimizer_values[i].value_count_per_parameter;
		cudaMalloc(&optimizer_values[i].values, sizeof(data_t) * len);
		cudaMemset(optimizer_values[i].values, 0, sizeof(data_t) * len);
	}
}

Optimizers::Optimizers()
{
	optimizer_values_per_parameter[Adam] = 7;
	optimizer_values_per_parameter[ElasticNet] = 0;

	for (size_t i = 0; i < last_optimizer_entry; i++)
		optimizer_values[i].value_count_per_parameter 
			= optimizer_values_per_parameter[i];
}

Optimizers::Optimizers(size_t parameter_count, optimizer_hyperparameters optimizer_options)
{
	*this = Optimizers();

	this->parameter_count = parameter_count;
	initialization_hyperparameters = optimizer_options;

	allocate_values();
	initialize_values(0, parameter_count);
}

Optimizers Optimizers::Clone()
{
	Optimizers out = Optimizers(parameter_count, initialization_hyperparameters);
	for (size_t i = 0; i < last_optimizer_entry; i++)
		copy_values_of(i, out);
	return out;
}

Optimizers::~Optimizers()
{
	for (size_t i = 0; i < optimizers_enum::last_optimizer_entry; i++)
	{
		cudaFree(optimizer_values[i].values);
		optimizer_values[i].values = 0;
	}
	
}

void Optimizers::save(FILE *file)
{
	save_array(&parameter_count, 1, file, false);
	save_array(&initialization_hyperparameters, 1, file, false);
	for (size_t i = 0; i < last_optimizer_entry; i++)
		save_values_of(i, file);
}

Optimizers Optimizers::load(FILE *file)
{
	Optimizers out = Optimizers();
	out.parameter_count = load_value<size_t>(file);
	out.initialization_hyperparameters = load_value<optimizer_hyperparameters>(file);
	for (size_t i = 0; i < last_optimizer_entry; i++)
		out.optimizer_values[i] = out.load_values_of(i, file);
	return out;
}

void Optimizers::set_initialization(optimizer_hyperparameters new_initialization)
{
	initialization_hyperparameters = new_initialization;
}

void Optimizers::add_parameters(size_t added_parameter_count, long insert_i)
{
	for (size_t i = 0; i < last_optimizer_entry; i++)
	{
		Optimizer_values values = optimizer_values[i];

		size_t values_per_parameter = values.value_count_per_parameter;
		if (!values_per_parameter) continue;

		size_t value_count = parameter_count * values_per_parameter;
		values.values = cuda_insert_zeros(
			values.values, value_count, insert_i * values_per_parameter, values_per_parameter * added_parameter_count, true
		);
	}
	parameter_count += added_parameter_count;
}

void Optimizers::remove_parameters(size_t removed_count, long removed_i)
{
	for (size_t i = 0; i < last_optimizer_entry; i++)
	{
		Optimizer_values values = optimizer_values[i];

		size_t values_per_parameter = values.value_count_per_parameter;
		if (values_per_parameter) continue;

		size_t value_count = parameter_count * values_per_parameter;
		values.values = cuda_remove_elements(
			values.values, value_count, removed_i * values_per_parameter, removed_count * values_per_parameter, true
		);
	}
	parameter_count -= removed_count;
}

__device__ void Optimizers::subtract_gradient(
	field_t *parameter, size_t parameter_i, data_t gradient,
	gradient_hyperparameters hyperparameters
)
{
	if (!parameter) return;

	if (hyperparameters.optimization.adam.active)
		gradient = apply_adam(gradient, optimizer_values[Adam], parameter_i);
	if (hyperparameters.optimization.L_regularization.active)
		gradient = apply_ElasticNet(*parameter, gradient, hyperparameters);
	gradient = apply_hyperparameters(gradient, hyperparameters);
	atomicAdd(parameter, -gradient);
}

__device__ data_t Optimizers::apply_hyperparameters(data_t gradient, gradient_hyperparameters hyperparameters)
{
	gradient *= hyperparameters.learning_rate;
	gradient = device_clip(gradient, -hyperparameters.gradient_clip, hyperparameters.gradient_clip);
	return gradient;
}

data_t Optimizers::apply_adam(data_t gradient, Optimizer_values values, size_t parameter_i)
{
	if (!values.values) throw;

   	size_t values_starting_i = values.value_count_per_parameter * parameter_i;

	data_t *optimizer_values = values.values;
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

	data_t new_gradient = bias_corrected_m / (sqrt(abs(bias_corrected_v)) + optimizer_values[values_starting_i + 2]);
	return (new_gradient);
}

__device__ data_t Optimizers::apply_ElasticNet(field_t parameter, data_t gradient, gradient_hyperparameters hyperparameters)
{
	data_t raw_L1_gradient = (parameter > 0) - (parameter < 0);
	data_t gradient_to_add = hyperparameters.optimization.L_regularization.alpha * raw_L1_gradient;
	
	data_t raw_L2_gradient = 2 * parameter;
	gradient_to_add += (1 - hyperparameters.optimization.L_regularization.alpha) * raw_L2_gradient;
	gradient_to_add *= hyperparameters.optimization.L_regularization.lambda;
	return gradient + gradient_to_add;
}
