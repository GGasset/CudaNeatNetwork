
#include "math.h"
#include "GAE.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

#include "NN_constructor.h"


template<typename t>
t abs(t a)
{
	return a * (-1 + 2 * (a > 0));
}

static void bug_hunting()
{
	const size_t input_len = 1;
	const size_t output_len = 2;

	const bool stateful = false;

	optimizer_hyperparameters optimizers;
	NN *n = NN_constructor()
		.append_layer(Dense, Neuron, 20, sigmoid)
		.append_layer(Dense, Neuron, 10, sigmoid)
		.append_layer(Dense, Neuron, output_len, sigmoid)
		.construct(input_len, optimizers, stateful);

	const size_t t_count = 2;
	data_t X[input_len * t_count];
	for (size_t i = 0; i < t_count; i++)
	{
		X[i] = i % 2 ? -.5 : .5;
	}

	data_t Y_hat[output_len * t_count];
	for (size_t i = 0; i < t_count; i++)
	{
		int odd = i % 2;
		Y_hat[i * output_len] = !odd ? .75 : .25;
		Y_hat[i * output_len + 1] = !odd ? .25 : .75;
	}

	gradient_hyperparameters hyperparameters;
	hyperparameters.learning_rate = .1;

	const size_t epochs = 6000;
	for (size_t i = 0; i < epochs || 1; i++)
	{
		data_t *Y = 0;
		printf("\n%i %.4f\n", i, n->training_batch(
			t_count,
			X, Y_hat, 1, output_len * t_count,
			MSE,
			&Y, host_cpp_pointer_output, hyperparameters
		));
		for (size_t j = 0; j < t_count; j++)
		{	
			for (size_t k = 0; k < output_len; k++) 
			{
				size_t output_index = j * output_len + k;
				printf("%i: %.2f  ", k, Y[output_index]);
			}
			printf("\n");
		}
		delete[] Y;
		printf("\n\n\n");
	}
}

// Test for LSTM neuron which consists in giving a positive or negative first input 
//		and testing its long term dependency of that input.
static void test_LSTM()
{
	const size_t in_len = 1;
	const size_t out_len = 2;

	optimizer_hyperparameters optimizer;
	NN* n = NN_constructor()
		.append_layer(Dense, LSTM, 32)
		.append_layer(Dense, Neuron, out_len)
		.construct(in_len, optimizer);

	const size_t t_count = 3;
	data_t X[in_len * t_count * 2] {};
	data_t Y_hat[out_len * t_count * 2] {};

	for (size_t i = 0; i < in_len * t_count; i++)
		X[i] = !i ? -.5 : 0;
	for (size_t i = 0; i < in_len * t_count; i++)
		X[in_len * t_count + i] = !i ? .5 : 0;

	for (size_t i = 0; i < t_count; i++)
	{
		Y_hat[out_len * i] = .75;
		Y_hat[out_len * i + 1] = .25;
	}
	for (size_t i = 0; i < t_count; i++)
	{
		Y_hat[out_len * t_count + out_len * i] = .25;
		Y_hat[out_len * t_count + out_len * i + 1] = .75;
	}

	gradient_hyperparameters hyperparameters;
	hyperparameters.optimization = optimizer;

	const size_t epoch_n = 5000;
	for (size_t i = 0; i < epoch_n; i++)
	{
		data_t* Y = 0;
		data_t* activations = 0;
		data_t* execution_values = 0;
		if (i % 1 == 0)
			printf("\n");
		for (size_t j = 0; j < 2; j++)
		{
			n->training_execute(
				t_count, 
				X + in_len * t_count * j, &Y, host_cpp_pointer_output,
				&execution_values, &activations
			);
			data_t cost = n->train(
				t_count,
				execution_values, activations, Y_hat + out_len * t_count * j, true, out_len * t_count,
				MSE, hyperparameters
			);

			if (i % 1 == 0)
				printf("%i | %.4f | %.4f, %.4f\n", i, cost, Y[out_len * t_count - 2], Y[out_len * t_count - 1]);

			delete[] Y;
			Y = 0;
		}
	}
}

static void NEAT_evolution_test()
{
	const size_t input_len = 50;
	const size_t output_len = 5;
	const size_t t_count = 1;

	/*data_t X[input_len * t_count]{};
	for (size_t i = 0; i < input_len * t_count; i++)
	{
		X[i] = .5;
	}*/

	/*data_t Y_hat[output_len * t_count]{};
	for (size_t i = 0; i < output_len * t_count; i++)
	{
		Y_hat[i] = .5;
	}*/

	optimizer_hyperparameters optimizer;
	NN* n = NN_constructor()
		//.append_layer(NEAT, Neuron, 1, sigmoid)
		.append_layer(NEAT, Neuron, 5, sigmoid)
		.append_layer(NEAT, Neuron, output_len, activations_last_entry)
		.construct(input_len, optimizer);
	
	gradient_hyperparameters hyperparameters;
	hyperparameters.learning_rate = .01;
	size_t epochs = 50000;
	
	for (size_t epoch = 0; epoch < epochs || !epoch; epoch++)
	{
		n->evolve();
		//if (epoch < 250 && 1) n->add_neuron(0);
		//if (epoch % 2 && 0) n->remove_neuron(0);

		data_t *X = (data_t*)calloc(n->get_input_length() * t_count, sizeof(data_t));
		for (size_t i = 0; i < n->get_input_length() * t_count; i++)
			X[i] = .5;
		data_t *Y_hat = (data_t*)calloc(n->get_output_length() * t_count, sizeof(data_t));
		for (size_t i = 0; i < n->get_output_length() * t_count; i++)
			Y_hat[i] = 35;

		data_t *Y = 0;
		//Y = n->inference_execute(X);
		n->training_batch(t_count, X, Y_hat, true, n->get_output_length() * t_count, MSE,
			&Y, host_cpp_pointer_output, hyperparameters);
		for (size_t i = 0; i < n->get_output_length() * t_count; i++)
			if (((cudaPeekAtLastError() != cudaSuccess) || Y[i] != Y[i]))
			{
				epochs = 0;
				printf("error");
				if (cudaGetLastError() != cudaSuccess) printf(" Because of cuda error\n");
				else printf(" %.2f\n", Y[i]);
				break;
			}
			else
				printf("%.2f ", Y[i]);
		printf("\n");

		free(X);
		free(Y_hat);
		delete[] Y;

		n->print_shape();
	}
}

static void test_PPO()
{
	const size_t input_len = 4;
	const size_t output_len = 4;

	optimizer_hyperparameters value_function_optimizer;
	NN* value_function = NN_constructor()
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 20, _tanh)
		.append_layer(Dense, Neuron, 1, _tanh)
		.construct(input_len, value_function_optimizer);

	optimizer_hyperparameters agent_optimizer;
	NN* agent = NN_constructor()
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 15)
		.append_layer(Dense, Neuron, 10)
		.append_layer(Dense, Neuron, output_len)
		.construct(input_len, agent_optimizer);

	PPO_hyperparameters parameters;
	parameters.max_training_steps = 20;
	parameters.GAE.training_steps = 5;
	//parameters.GAE.gamma = .99;
	parameters.GAE.value_function.learning_rate = 1e-2;
	parameters.policy.learning_rate = 1e-3;
	parameters.clip_ratio = .2;
	parameters.max_kl_divergence_threshold = .01;

	parameters.policy.regularization.entropy_bonus.active = true;
	parameters.policy.regularization.entropy_bonus.entropy_coefficient = 1E-3;

	const size_t board_side_len = 3;
	const size_t board_square_count = board_side_len * board_side_len;
	const size_t max_t_count = board_side_len + 5;

	size_t total_frames = 0;
	for (size_t epoch = 0; true; epoch++)
	{
		data_t *initial_states = 0;
		data_t *trajectory_inputs = 0;
		data_t *trajectory_outputs = 0;

		int agent_position_i = rand() % board_square_count;
		int goal_i = rand() % (board_square_count - 1);
		goal_i += goal_i >= agent_position_i;
		
		bool achieved_objective = false;

		data_t rewards[max_t_count]{};
		bool finished = false;
		size_t i;
		for   (i = 0; i < max_t_count && !finished; i++, total_frames++)
		{
			int delta_y = goal_i / board_side_len - agent_position_i / board_side_len;
			int delta_x = goal_i % board_side_len - agent_position_i % board_side_len;

			int norm_delta_y = delta_y > 0;
			norm_delta_y -= !norm_delta_y;
			int norm_delta_x = delta_x > 0;
			norm_delta_x -= !norm_delta_x;

			data_t X[input_len] {};
			X[0] = norm_delta_x;
			X[1] = -norm_delta_x;
			X[2] = norm_delta_y;
			X[3] = -norm_delta_y;

			data_t *Y = 0;
			Y = agent->PPO_execute(
				X,
				&initial_states, &trajectory_inputs, &trajectory_outputs,
				i
			);

			data_t x_probs_magnitude = Y[0] + Y[1];
			data_t left_probability = Y[0];
			data_t right_probability = Y[1];
			if (x_probs_magnitude > 1)
			{
				left_probability /= x_probs_magnitude;
				right_probability /= x_probs_magnitude;
			}
			data_t r = get_random_float();
			int x_action = 0;
			x_action -= r < left_probability;
			x_action += r < left_probability + right_probability && !x_action;

			data_t y_probs_magnitude = Y[2] + Y[3];
			data_t down_probability = Y[2];
			data_t up_probability = Y[3];
			if (y_probs_magnitude > 1)
			{
				down_probability /= y_probs_magnitude;
				up_probability /= y_probs_magnitude;
			}
			r = get_random_float();
			int y_action = 0;
			y_action -= r < down_probability;
			y_action += r < down_probability + up_probability && !y_action;

			int agent_updated_position = agent_position_i + x_action + board_side_len * y_action;
			/*if (agent_position_i == 10)
			{
				printf("Now ");
			}*/
			if (
				agent_position_i / board_side_len != (agent_position_i + x_action) / board_side_len
				|| 
				agent_updated_position < 0 || agent_updated_position / board_side_len >= board_side_len
			)
			{ // Agent out of bounds
				rewards[i] = -.7;
				finished = true;
				//printf("%i %i Out of bounds ", agent_position_i / board_side_len != (agent_position_i + x_action) / board_side_len, agent_position_i);
			}

			if (agent_updated_position == goal_i)
			{
				rewards[i] = .7;
				finished = true;
				achieved_objective = true;
			}
			else if (i == max_t_count - 1)
			{
				rewards[i] = -.5;
			}

			agent_position_i = agent_updated_position;
			delete[] Y;
		}

		agent->PPO_train(
			i, &initial_states, &trajectory_inputs, &trajectory_outputs,
			rewards, true, value_function, parameters
		);

		printf("%i, %i, %zi, %zi\n", (int)achieved_objective, i, epoch, total_frames);
	}
	delete value_function;
	delete agent;
}

int main()
{
#ifdef DETERMINISTIC
	srand(13);
#else
	srand(get_arbitrary_number());
#endif


	//cudaSetDevice(0);
	//bug_hunting();
	//test_LSTM();
	test_PPO();

	printf("Last error peek: %i\n", cudaPeekAtLastError());
}
