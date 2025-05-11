
#include "Optimizer_init.h"

__device__ IOptimizer* initialize_optimizer(optimizers_enum optimizer, size_t parameter_count)
{
	IOptimizer* out = 0;

	switch (optimizer)
	{
	case no_optimizer:
		cudaMalloc(&out, sizeof(IOptimizer));
		*out = IOptimizer();
		out->alloc_optimizer_values(parameter_count, false);
		break;
	case Adam:
		cudaMalloc(&out, sizeof(AdamOptimizer));
		*out = AdamOptimizer();
		out->alloc_optimizer_values(parameter_count, false);
		break;
	default:
		return (0);
	}
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
