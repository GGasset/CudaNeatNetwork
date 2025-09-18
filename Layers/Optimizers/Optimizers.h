
#pragma once

#include <cstddef>

#include "data_type.h"
#include "NN_enums.h"
#include "gradient_parameters.h"


typedef struct Optimizer_values
{
	data_t  *values;
	size_t  value_count_per_parameter;
};

class Optimizers
{
private:
	Optimizer_values optimizer_values[optizers_enum::last_optimizer_entry];
	size_t parameter_count;
	
public:	
	Optimizers();
	Optimizers(optimizers_hyperparameters optimizer_options);
	~Optimizers();
};
