
#include "Optimizer_init.h"

__device__ IOptimizer* initialize_optimizer(optimizers_enum optimizer)
{
	IOptimizer* out = 0;

	switch (optimizer)
	{
	case no_optimizer:
		cudaMalloc(&out, sizeof(IOptimizer));
		break;
	case Adam:
		break;
	default:
		return (0);
	}
	return (out);
}

__global__ void global_optimizer_init(optimizers_enum optimizer, IOptimizer** out)
{
	*out = initialize_optimizer(optimizer);
}

__host__ IOptimizer* host_optimizer_init(optimizers_enum optimizer)
{
	IOptimizer** tmp = 0;
	cudaMalloc(&tmp, sizeof(IOptimizer*));
	global_optimizer_init kernel(1, 1) (optimizer, tmp);
	cudaDeviceSynchronize();

	IOptimizer* out = 0;
	cudaMemcpy(&out, tmp, sizeof(IOptimizer*), cudaMemcpyDeviceToHost);
	return (out);
}
