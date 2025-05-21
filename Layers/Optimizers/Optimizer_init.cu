
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
}

__global__ void get_optimizer_data_buffer(IOptimizer* optimizer, void** out_buffer, size_t *buff_len)
{
	if (!out_buffer || !buff_len) return;

	size_t values_per_paramater = optimizer->values_per_parameter;
	size_t param_count = optimizer->parameter_count;
	size_t value_count = values_per_paramater * param_count;

	size_t header_size = sizeof(optimizers_enum) + sizeof(size_t) * 2;
	size_t buff_size = header_size + sizeof(field_t) * value_count;
	char* out = 0;
	cudaMalloc(&out, buff_size);
	if (!out) return;

	memcpy(out, optimizer, header_size);
	if (optimizer->optimizer_values && value_count)
		memcpy(out + header_size, optimizer->optimizer_values, sizeof(field_t) * value_count);

	*out_buffer = out;
	*buff_len = buff_size;
}

__host__ void host_save_optimizer(FILE* file, IOptimizer* optimizer)
{
	void** device_buff = 0;
	size_t* device_buff_len = 0;

	cudaMalloc(&device_buff, sizeof(char *));
	cudaMalloc(&device_buff_len, sizeof(size_t));
	
	get_optimizer_data_buffer kernel(1, 1) (optimizer, device_buff, device_buff_len);
	cudaDeviceSynchronize();

	size_t buff_len = 0;
	cudaMemcpy(&buff_len, device_buff_len, sizeof(size_t), cudaMemcpyDeviceToHost);
	cudaFree(device_buff_len);

	void* buff = new char[buff_len];
	cudaMemcpy(buff, device_buff, buff_len, cudaMemcpyDeviceToHost);
	fwrite(buff, 1, buff_len, file);
	delete[] buff;
}
