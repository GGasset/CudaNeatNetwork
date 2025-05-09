
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"
#include "NN_enums.h"
#include "cuda_functionality.cuh"
#include "functionality.h"
#include "gradient_parameters.h"

#pragma once

// Must be created inside device code
class IOptimizer
{
public:
	size_t values_per_parameter = 0;
	size_t parameter_count = 0;
	field_t* optimizer_values = 0;

	inline IOptimizer() {};
	__device__ void alloc_optimizer_values(size_t param_count, bool copy_old_values);
	__device__ virtual void initialize_optimizer_values(field_t* values);
	__device__ void cleanup();
	
	__device__ void hyperparameter_subtract_gradient(field_t *parameter, data_t gradient, size_t layer_parameter_i, gradient_hyperparameters hyperparameters);
	/// <param name="layer_parameter_i">Basically gradient i</param>
	__device__ virtual void subtract_gradient(field_t *parameter, data_t gradient, size_t layer_parameter_i, gradient_hyperparameters hyperparameters);
};