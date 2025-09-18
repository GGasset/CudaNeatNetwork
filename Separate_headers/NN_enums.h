#include "data_type.h"

#ifndef ENUMS_H
#define ENUMS_H

enum NeuronTypes : size_t
{
	Neuron,
	LSTM,
	last_neuron_entry
};

enum ConnectionTypes : size_t
{
	Dense,
	NEAT,
	last_connection_entry
};

enum output_pointer_type : size_t
{
	no_output,
	cuda_pointer_output,
	host_cpp_pointer_output
};

enum LearningRateAdjusters : size_t
{
	high_learning_high_learning_rate,
	high_learning_low_learning_rate,
	cost_times_learning_rate,
	none
};

enum ActivationFunctions : size_t
{
	sigmoid,
	_tanh,
	activations_last_entry,
	no_activation
};

enum CostFunctions : size_t
{
	MSE,
	log_likelyhood,
};

enum optimizers_enum : size_t
{
	Adam,
	ElasticNet,
	last_optimizer_entry,
	no_optimizer
};

enum action_enum : size_t
{
	construct,
	destruct,
	save,
	load,
	last_entry
};

typedef struct {
	data_t *return_value;
	size_t value_count;
	size_t error;
} return_specifier;

enum error_values : size_t
{
	OK,
	NN_not_found
};

#endif
