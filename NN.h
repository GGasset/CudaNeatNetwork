#pragma once

#include <stdio.h>

#include "kernel_macros.h"
#include "functionality.h"

#include "NN_enums.h"
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
#include "GAE.cuh"

class NN
{
private:
	ILayer **layers = 0;
	size_t layer_count = 0;
	size_t neuron_count = 0;
	size_t input_length = 0;
	size_t output_length = 0;
	size_t* output_activations_start = 0;
	size_t execution_value_count = 0;
	size_t derivative_count = 0;
	short contains_recurrent_layers = 0;
	size_t gradient_count = 0;

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
	bool   is_recurrent();

	~NN();
	NN(ILayer** layers, size_t input_length, size_t layer_count);

	void execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, output_pointer_type output_type);
	void set_up_execution_arrays(data_t** execution_values, data_t** activations, size_t t_count);
	data_t* batch_execute(data_t* input, size_t t_count, output_pointer_type output_type = host_cpp_pointer_output);
	data_t* inference_execute(data_t* input, output_pointer_type output_type = host_cpp_pointer_output);

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
		output_pointer_type output_type,
		gradient_hyperparameters hyperparameters
	);

	/// <summary>
	/// Execution function made for training data generation
	/// </summary>
	void training_execute(
		size_t t_count,
		data_t* X,
		data_t** Y,
		output_pointer_type output_type,
		data_t** execution_values,
		data_t** activations,
		size_t arrays_t_length = 0
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

	/// <summary>
	/// Inference function to be used before calling PPO_train
	/// PPO_train deletes the arrays generated during the calls to this function
	/// Don't modify input/output neuron count between execution of this function, it is ok right after PPO_train and before this function
	/// </summary>
	/// <param name="X">Input</param>
	/// <param name="initial_states">Saves the hidden states before its first execution, create a pointer variable set to zero and pass its pointer</param>
	/// <param name="trajectory_inputs">Saves all inputs passed, create a pointer variable set to zero and pass its pointer</param>
	/// <param name="trayectory_outputs">Saves all inputs passed, create a pointer variable set to zero and pass its pointer</param>
	/// <param name="n_executions">Contains the number of times this function has been called after the last PPO_train call</param>
	/// <returns>Network output, 0 on error</returns>
	data_t* PPO_execute(data_t *X, data_t **initial_states, data_t **trajectory_inputs, data_t **trayectory_outputs, size_t n_executions);
	void PPO_train(
		size_t t_count,
		data_t **initial_states, data_t **trajectory_inputs, data_t **trajectory_outputs,
		data_t* rewards, bool are_rewards_at_host, NN *value_function_estimator,
		PPO_hyperparameters hyperparameters
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
