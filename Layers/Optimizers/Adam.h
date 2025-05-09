
#include "IOptimizer.h"

#pragma once
class AdamOptimizer : public IOptimizer
{
public:
	// Can be called inside device code
	AdamOptimizer();

	__device__ virtual void initialize_optimizer_values(field_t* values) override;
	__device__ virtual void subtract_gradient(field_t* parameter, data_t gradient, size_t layer_parameter_i);
};
