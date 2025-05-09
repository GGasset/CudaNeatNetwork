
#include "Adam.h"

AdamOptimizer::AdamOptimizer()
{
	values_per_parameter = 4;
}

void AdamOptimizer::initialize_optimizer_values(field_t* values)
{
	field_t tmp[4]{};
	tmp[0] = 0.001;
	tmp[1] = 0.9;
	tmp[2] = 0.999;
	tmp[3] = 10E-8;
	for (size_t i = 0; i < values_per_parameter; i++)
		cudaMemcpy(values + 4 * i, tmp, sizeof(field_t) * 4, cudaMemcpyDeviceToDevice);
}

__device__ void AdamOptimizer::subtract_gradient(field_t* parameter, data_t gradient, size_t layer_parameter_i)
{
	return __device__ void();
}
