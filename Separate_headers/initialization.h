
#pragma once

#include "data_type.h"

enum initialization_type
{
	Xavier,
	central_limit,
	constant
};

struct central_limit_initialization_parameters
{
	data_t mean = 0;
	data_t std = 1;
};

struct constant_initialization_parameters
{
	data_t value_constant = 0;
};

struct initialization_parameters
{
	initialization_type initalization = Xavier;

	central_limit_initialization_parameters central_limit;
	constant_initialization_parameters constant;

	// Automatically set in nn_constructor
	size_t layer_n_inputs = 0;
	// Automatically set in nn_constructor
	size_t layer_n_outputs = 0;
	// Automatically set in nn_constructor
	size_t time = 0;
};
