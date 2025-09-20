
#pragma once

#include <cstddef>
#include <cstdio>

#include "data_type.h"
#include "NN_enums.h"
#include "gradient_parameters.h"

#include "cuda_functionality.cuh"

// Holds variables to share between calls to subtract gradients, in which optimization hyperparameters is passed.
typedef struct
{
	data_t  *values = 0;
	size_t  value_count_per_parameter = 0;
} Optimizer_values;

class Optimizers
{	
private:
	/*
	Steps to add an optimizer:

	Add it to optimizers enum before last optimizer_entry and without a set value
	create the function that applies the optimizer
	Add its value_count per parameter to Optimizers default constructor
	use the function in subtract_gradient
	*/
	Optimizer_values optimizer_values[last_optimizer_entry];
	int optimizer_values_per_parameter[last_optimizer_entry];
	size_t parameter_count = 0;
	optimizer_hyperparameters initialization_hyperparameters;

	// Does not allocate values
	void copy_values_of(size_t optimizer_i, Optimizers dst);
	
	void save_values_of(size_t optimizer_i, FILE *file);
	Optimizer_values load_values_of(size_t optimizer_i, FILE *file);

	void initialize_values(long parameter_start_i, size_t parameter_count);
	void allocate_values();

public:
	Optimizers();
	// optimizer_options is used for hyperparameter initialization
	Optimizers(size_t parameter_count, optimizer_hyperparameters optimizer_options);
	Optimizers Clone();
	~Optimizers();

	void save(FILE *file);
	static Optimizers *load(FILE *file);

	// sets the hyperparameters that will be used for initialization
	void set_initialization(optimizer_hyperparameters new_initialization);

	// Adds values to fit new parameters, also initializes them
	//
	// block_start_i: the start position of the inserted parameters, < 0 to append (i.e. -1)
	void add_parameters(size_t added_parameter_count, long insert_i);

	// Removes values to adjust for deleted parameters
	//
	// block_start_i: the start position of the deleted parameters, < 0 to remove from the end (i.e. -1)
	void remove_parameters(size_t removed_count, long removed_i);

	__device__ void subtract_gradient(
		field_t *parameter, size_t parameter_i, data_t gradient,
		gradient_hyperparameters hyperparameters
	);
	
	// Applies learning rate and gradient clip
	__device__ data_t apply_hyperparameters(data_t gradient, gradient_hyperparameters hyperparameters);
	// Returns new gradient after applying adam
	__device__ data_t apply_adam(data_t gradient, Optimizer_values values, size_t parameter_i);
	__device__ data_t apply_ElasticNet(field_t parameter, data_t gradient, gradient_hyperparameters hyperparameters);
};

