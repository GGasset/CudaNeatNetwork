#pragma once

#include <stdio.h>

#include "kernel_macros.h"
#include "functionality.h"

#include "NN_enums.h"
#include "nn_lens.h"
#include "neuron_operations.cuh"
#include "costs.cuh"
#include "gradient_parameters.h"
#include "RL_parameters.h"

#include "DenseConnections.h"
#include "NeatConnections.h"
#include "NeuronLayer.h"
#include "LSTMLayer.h"

#include "regularization.cuh"
#include "Optimizers.h"
//#include "GAE.cuh"

class NN
{
private:
	ILayer **layers = 0;
	size_t input_length = 0;
	size_t output_length = 0;
	size_t* output_activations_start = 0;
	short contains_recurrent_layers = 0;
	
	nn_lens counts{};

	initialization_parameters weight_init;
	initialization_parameters bias_init;
	initialization_parameters layer_weights_init;

protected:
	void set_fields();

public:
	NN();

	evolution_metadata evolution_values;
	optimizer_hyperparameters optimizer_initialization;
	bool stateful = false;

	size_t get_input_length();
	size_t get_neuron_count();
	size_t get_output_length();
	size_t get_output_activations_start();
	size_t get_gradient_count_per_t();
	bool   is_recurrent();

	~NN();
	NN(
		ILayer** layers, size_t input_length, size_t layer_count,
		initialization_parameters weight_init = {.initialization=Xavier},
		initialization_parameters bias_init = {.initialization=constant},
		initialization_parameters layer_weight_init = {.initialization=Xavier}
	);

	// ## Notes:
	// If the array lengths don't match with the parameters, the function call is ignored, and null is returned
	// For the socket, create a wrapper at the User class level, to abstract arr_location, for example.
	//
	// The activations and execution values are automatically copied to add space for the new execution at each execution line
	// At a higher level, just increment t_count_per_execution_line
	// No input size of activations and execution values is requested as they are internal arrays that won't be passed to the socket
	// A wrapper is highly incentivized
	//
	// ---
	// ## Params:
	// - execution_lines: 
	//   The number of executions to execute at the same time
	//   Used, for example, in PPO that different environments are executed simultaniously
	//   Or, for executing all the batch at the same time
	//   In recurrent layers, different execution lines won't share the same state/context
	//   Only the last t of each execution line will actually be executed
	// ---
	// - t_count_per_execution_line:
	//   The number of already executed timesteps inside each execution line, they will be ignored, excluding the last one of each line
	//   If not null, each array should start with the given number of timesteps of values
	// ---
	// - prev_execution_values:
	//   Used for setting the state of the new recurrent layers, if the network is not recurrent, they are ignored
	//
	// ---
	// ## Returns:
	// - Y in the location specified at the parameter arr_location output_type
	std::tuple<data_t *>execute(
		size_t execution_lines, size_t t_count_per_execution_line,
		data_t *X, size_t X_len, arr_location output_type,
		data_t **activations, data_t **execution_values,
		bool delete_memory_before = false,
		data_t *prev_execution_values = 0, size_t prev_execution_values_len = 0
	);

	// ## Summary: 
	//  - Calculates the gradients for all the timesteps of all the executions of the execution_lines
	//  - If the array lengths don't match with the parameters, the function call is ignored, and null is returned
	// ---
	// 
	// ## Params:
	// - execution_lines: 
	//   The number of executions to execute at the same time
	//   Used, for example, in PPO that different environments are executed simultaniously
	//   Or, for executing all the batch at the same time
	//   In recurrent layers, different execution lines won't share the same state/context
	// ---
	// - t_count_per_execution_line:
	//   The number of already executed timesteps inside each execution line, they will be ignored, excluding the last one of each line
	//   If not null, each array should start with the given number of timesteps of values
	// ---
	// ## Returns:
	// - Gradients (device arr), mean error
	std::tuple<data_t *, data_t> backpropagate(
		size_t execution_lines, size_t t_count_per_execution_line,
		data_t *activations, size_t activations_len,
		data_t *execution_values, size_t execution_values_len,
		gradient_hyperparameters
	);
	
	// Subtracts all the gradients from all execution lines parallely through gradient accumulation before subtracting
	void subtract_gradients(
		size_t execution_lines, size_t t_count_per_execution_line, 
		data_t *gradients, size_t gradients_len,
		gradient_hyperparameters hyperparameters
	);

	// Training and execution are about to be deprecated
	
	void execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, arr_location output_type);
	void set_up_execution_arrays(data_t** execution_values, data_t** activations, size_t t_count);
	data_t* batch_execute(data_t* input, size_t t_count, arr_location output_type = host_arr_new);
	data_t* inference_execute(data_t* input, arr_location output_type = host_arr_new);

	data_t adjust_learning_rate(
		data_t learning_rate,
		data_t cost,
		LearningRateAdjusters adjuster,
		data_t max_learning_rate,
		data_t previous_cost = 0
	);

	data_t training_batch(
		size_t t_count,
		data_t* X,
		data_t* Y_hat,
		bool is_Y_hat_on_host_memory,
		size_t Y_hat_value_count,
		CostFunctions cost_function,
		data_t** Y,
		arr_location output_type,
		gradient_hyperparameters hyperparameters
	);

	/// <summary>
	/// Execution function made for training data generation
	/// </summary>
	void training_execute(
		size_t t_count,
		data_t* X,
		data_t** Y,
		arr_location output_type,
		data_t** execution_values,
		data_t** activations,
		size_t arrays_t_length = 0,
		std::vector<bool> *delete_mem = 0
	);

	data_t train(
		size_t t_count,
		data_t* execution_values,
		data_t* activations,
		data_t* Y_hat,
		bool is_Y_hat_on_host_memory,
		size_t Y_hat_value_count,
		CostFunctions cost_function,
		gradient_hyperparameters hyperparameters
	);

	data_t calculate_output_costs(
		CostFunctions cost_function,
		size_t t_count,
		data_t* Y_hat,
		data_t* activations, size_t activations_start,
		data_t* costs, size_t costs_start
	);

	/// <param name="gradients">- pointer to cero and to a valid array are valid</param>
	void backpropagate(
		size_t t_count,
		data_t* costs,
		data_t* activations,
		data_t* execution_values,
		data_t** gradients,
		gradient_hyperparameters hyperparameters
	);

	// Automatically called inside NN::backpropagate()
	void apply_regularizations(
		size_t t_count,
		data_t *costs, data_t *activations,
		regularization_hyperparameters hyperparameters
	);

	void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
		data_t* execution_values, size_t execution_values_start
	);

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start,
		data_t* costs, size_t costs_start,
		data_t* gradients, size_t gradients_start, size_t next_gradients_start,
		data_t* derivatives, size_t derivatives_start, size_t previous_derivatives_start,
		float dropout_rate
	);

	void subtract_gradients(
		data_t* gradients, size_t gradients_start, gradient_hyperparameters hyperparameters
	);

	data_t* get_hidden_state(size_t *arr_value_count = 0);
	void set_hidden_state(data_t *state, int free_input_state);

	void evolve();
	void add_layer();
	void add_layer(size_t insert_i);
	void add_layer(size_t insert_i, NeuronTypes layer_type);
	void add_layer(size_t insert_i, ILayer* layer);
	void add_output_neuron();
	void add_input_neuron();
	void add_neuron(size_t layer_i);
	
	/// <param name="neuron_i">in respect to the whole network</param>
	void adjust_to_added_neuron(int layer_i, size_t neuron_i);
	void remove_neuron(size_t layer_i);
	void remove_neuron(size_t layer_i, size_t layer_neuron_i);


	void delete_memory();

	NN* clone();
	void save(const char *pathname);
	void save(FILE* file);
	static NN* load(const char *pathname, bool load_state = true);
	static NN* load(FILE* file);

	void deallocate();

	void print_shape();
};
