
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
	global_optimizer_init kernel(1, 1) (optimizer, tmp, parameter_count);
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
	delete optimizer;
}

__global__ void call_optimizer_values_alloc(IOptimizer* optimizer, size_t new_param_count, bool copy_old_values)
{
	if (get_tid() || !optimizer) return;

	optimizer->alloc_optimizer_values(new_param_count, copy_old_values);
}

__global__ void get_optimizer_data_buffer(IOptimizer* optimizer, char* out_buffer, size_t out_buffer_size, size_t *buff_len)
{
	if (get_tid()) return;

	size_t values_per_paramater = optimizer->values_per_parameter;
	size_t param_count = optimizer->parameter_count;
	size_t value_count = values_per_paramater * param_count;

	size_t header_size = sizeof(optimizers_enum) + sizeof(size_t) * 2;
	size_t buff_size = header_size + sizeof(field_t) * value_count;

	if (buff_len)
		*buff_len = buff_size;

	if (out_buffer && out_buffer_size >= buff_size)
	{
		*(optimizers_enum*)out_buffer = optimizer->optimizer;
		*(size_t*)(out_buffer + sizeof(optimizers_enum)) = optimizer->values_per_parameter;
		*(size_t*)(out_buffer + sizeof(optimizers_enum) + sizeof(size_t)) = optimizer->parameter_count;
		if (optimizer->optimizer_values && value_count)
			memcpy(out_buffer + header_size, optimizer->optimizer_values, sizeof(field_t) * value_count);
	}
}

__host__ void host_save_optimizer(FILE* file, IOptimizer* optimizer)
{
	if (!optimizer) return;

	size_t *device_buff_len = 0;
	cudaMalloc(&device_buff_len, sizeof(size_t));
	get_optimizer_data_buffer kernel(1, 1) (optimizer, 0, 0, device_buff_len);
	cudaDeviceSynchronize();

	size_t buff_len = 0;
	cudaMemcpy(&buff_len, device_buff_len, sizeof(size_t), cudaMemcpyDeviceToHost);
	cudaFree(device_buff_len);

	char* device_buffer = 0;
	cudaMalloc(&device_buffer, buff_len);
	cudaMemset(device_buffer, 0, buff_len);
	get_optimizer_data_buffer kernel(1, 1) (optimizer, device_buffer, buff_len, 0);
	cudaDeviceSynchronize();

	char* buffer = new char[buff_len];
	cudaMemcpy(buffer, device_buffer, buff_len, cudaMemcpyDeviceToHost);
	cudaFree(device_buffer);

	fwrite(buffer, 1, buff_len, file);
	delete[] buffer;
}

static __global__ void set_optimizer_values(IOptimizer *optimizer, field_t *values, size_t value_count)
{
	size_t tid = get_tid();
	if (tid >= value_count || !optimizer || !optimizer->optimizer_values || tid >= optimizer->parameter_count * optimizer->values_per_parameter) return;

	optimizer->optimizer_values[tid] = values[tid];
}

__host__ IOptimizer* host_load_optimizer(FILE *file)
{
	size_t header_size = sizeof(optimizers_enum) + sizeof(size_t);

	optimizers_enum optimizer = no_optimizer;
	size_t values_per_parameter = 0;
	size_t parameter_count = 0;

	fread(&optimizer, sizeof(optimizers_enum), 1, file);
	fread(&values_per_parameter, sizeof(size_t), 1, file);
	fread(&parameter_count, sizeof(size_t), 1, file);

	IOptimizer* out = host_optimizer_init(optimizer, parameter_count);

	size_t value_count = values_per_parameter * parameter_count;
	field_t* values = new field_t[value_count];
	fread(values, sizeof(field_t), value_count, file);

	field_t* device_values = 0;
	cudaMalloc(&device_values, sizeof(field_t) * value_count);
	cudaMemcpy(device_values, values, sizeof(field_t) * value_count, cudaMemcpyHostToDevice);
	delete[] values;

	set_optimizer_values kernel(value_count / 32 + (value_count % 32 > 0), 32) (out, device_values, value_count);
	cudaDeviceSynchronize();
	cudaFree(device_values);
	return (out);
}

static __global__ void clone_optimizer(IOptimizer* to_clone, IOptimizer** out, size_t* optimizer_value_count)
{
	if (!to_clone || !optimizer_value_count || !out || get_tid()) return;

	optimizers_enum optimizer_type = to_clone->optimizer;

	IOptimizer* cloned = initialize_optimizer(optimizer_type, to_clone->parameter_count);
	if (!cloned) return;

	*out = cloned;
	*optimizer_value_count = cloned->parameter_count * cloned->values_per_parameter;
}

static __global__ void clone_optimizer_values(IOptimizer *src, IOptimizer *optimizer, size_t optimizer_value_count)
{
	size_t tid = get_tid();
	if (tid >= optimizer_value_count) return;

	optimizer->optimizer_values[tid] = src->optimizer_values[tid];
}

__host__ IOptimizer* host_clone_optimizer(IOptimizer* to_clone)
{
	if (!to_clone) return 0;

	IOptimizer** cloned_out_pntr = 0;
	cudaMalloc(&cloned_out_pntr, sizeof(IOptimizer*));
	cudaMemset(cloned_out_pntr, 0, sizeof(IOptimizer*));

	size_t* optimizer_value_count_pntr = 0;
	cudaMalloc(&optimizer_value_count_pntr, sizeof(size_t));
	cudaMemset(optimizer_value_count_pntr, 0, sizeof(size_t));

	clone_optimizer kernel(1, 1) (to_clone, cloned_out_pntr, optimizer_value_count_pntr);
	cudaDeviceSynchronize();

	IOptimizer* cloned = 0;
	cudaMemcpy(&cloned, cloned_out_pntr, sizeof(IOptimizer *), cudaMemcpyDeviceToHost);
	cudaFree(cloned_out_pntr);

	size_t optimizer_value_count = 0;
	cudaMemcpy(&optimizer_value_count, optimizer_value_count_pntr, sizeof(size_t), cudaMemcpyDeviceToHost);
	cudaFree(optimizer_value_count_pntr);

	clone_optimizer_values kernel(optimizer_value_count / 32 + (optimizer_value_count % 32 > 0), 32) (
		to_clone, cloned, optimizer_value_count
	);
	cudaDeviceSynchronize();
	return cloned;
}
