
#include "Optimizer_init.h"

__device__ IOptimizer* initialize_optimizer(optimizers_enum optimizer, size_t parameter_count)
{
	IOptimizer* out = 0;

	switch (optimizer)
	{
	case no_optimizer:
		out = new IOptimizer();
		break;
	case Adam:
		out = new AdamOptimizer();
		break;
	default:
		return (0);
	}
	out->alloc_optimizer_values(parameter_count, false);
	return (out);
}

__global__ void global_optimizer_init(optimizers_enum optimizer, IOptimizer** out, size_t parameter_count)
{
	*out = initialize_optimizer(optimizer, parameter_count);
}

__host__ IOptimizer* host_optimizer_init(optimizers_enum optimizer, size_t parameter_count)
{
	IOptimizer** tmp = 0;
	cudaMalloc(&tmp, sizeof(IOptimizer*));
	global_optimizer_init <<<1, 1>>> (optimizer, tmp, parameter_count);
	cudaDeviceSynchronize();

	IOptimizer* out = 0;
	cudaMemcpy(&out, tmp, sizeof(IOptimizer*), cudaMemcpyDeviceToHost);
	cudaFree(tmp);
	return (out);
}

__global__ void call_Optimizer_destructor(IOptimizer *optimizer)
{
	if (!optimizer)
		return ;
	optimizer->cleanup();
}

__global__ void get_optimizer_data_buffer(IOptimizer* optimizer, void** out_buffer)
{
	if (!out_buffer) return;

	size_t values_per_paramater = optimizer->values_per_parameter;
	size_t param_count = optimizer->parameter_count;
	size_t value_count = values_per_paramater * param_count;

	size_t header_size = sizeof(optimizers_enum) + sizeof(size_t) * 2;
	size_t buff_size = header_size + sizeof(field_t) * value_count;
	char* out = new char[buff_size];
	if (!out) return;

	memcpy(out, optimizer, header_size);
	if (optimizer->optimizer_values && value_count)
		memcpy(out + header_size, optimizer->optimizer_values, sizeof(field_t) * value_count);

	*out_buffer = out;
}
