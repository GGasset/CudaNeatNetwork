#include "ILayer.h"
#include "costs.cuh"

#pragma once
class NN
{
private:
	ILayer **layers = 0;
	size_t max_layer_count = 0;
	size_t layer_count = 0;
	size_t neuron_count = 0;
	size_t input_length = 0;
	size_t output_length = 0;
	size_t* output_activations_start = 0;
	size_t execution_value_count = 0;
	size_t derivative_count = 0;
	short contains_recurrent_layers = 0;
	size_t gradient_count = 0;

	// Now state derivatives are 1 (variable) by default so no need to save derivatives
	//data_t* activations_since_memory_deletion = 0;
	//data_t* execution_values_since_memory_deletion = 0;
	//data_t* derivatives_since_memory_deletion = 0;
	//size_t since_memory_deletion_t_count = 0;

protected:
	void set_fields();

public:
	~NN();
	NN(short contains_recurrent_layers, ILayer** layers, size_t input_length, size_t layer_count, size_t max_layer_count = 0);
	
	void execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, short copy_output_to_host);
	data_t* execute(data_t* input, size_t t_count);
	data_t* execute(data_t* input);

	void calculate_supervised_output_costs(
		CostFunctions cost_function,
		size_t t_count,
		data_t* Y_hat,
		data_t* activations, size_t activations_start,
		data_t* costs, size_t costs_start
	);

	void train(
		bool stateful,
		size_t t_count,
		data_t* costs,
		data_t* gradients,
		data_t* derivatives,
		data_t* activations,
		data_t* execution_values
	);

	void alloc_training_variables(
		size_t t_count,
		data_t** costs,
		data_t** gradients,
		data_t** derivatives
	);

	void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start
	);

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start,
		data_t* costs, size_t costs_start,
		data_t* gradients, size_t gradients_start, size_t next_gradients_start,
		data_t* derivatives, size_t derivatives_start, size_t previous_derivatives_start,
		short calculate_derivatives
	);

	void subtract_gradients(
		data_t* gradients, size_t gradients_start
	);

	void deallocate();
};

