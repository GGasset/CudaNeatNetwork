
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"
#include "NN_enums.h"
#include "cuda_functionality.cuh"
#include "functionality.h"
#include "gradient_parameters.h"

class Optimizer_device
{
public:
	size_t values_per_parameter;
	size_t parameter_count;
	field_t* optimizer_values;

	Optimizer_device();
	~Optimizer_device();

	void alloc_optimizer_values(size_t param_count, bool copy_old_values);
	virtual void initialize_optimizer_values(field_t* values);
	
	__device__ void hyperparameter_subtract_gradient(field_t *parameter, data_t gradient, size_t layer_parameter_i, gradient_hyperparameters hyperparameters);
	/// <param name="layer_parameter_i">Basically gradient i</param>
	__device__ virtual void subtract_gradient(field_t *parameter, data_t gradient, size_t layer_parameter_i);
};

class Optimizer_host
{
public:
	Optimizer_host();
	~Optimizer_host();

	/// <summary>
	/// Must be created inside device code
	/// </summary>
	Optimizer_device *optimizer;
private:

};