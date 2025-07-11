﻿
#include "math.h"
#include "GAE.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <sstream>

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
	NN *n = NN_constructor()
		.append_layer(Dense, Neuron, 20, sigmoid)
		.append_layer(Dense, Neuron, 10, sigmoid)
		.append_layer(Dense, Neuron, output_len, sigmoid)
		.construct(input_len, Adam, stateful);

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

	const size_t epochs = 6000;
	for (size_t i = 0; i < epochs || 1; i++)
	{
		data_t *Y = 0;
		printf("\n%i %.4f\n", i, n->training_batch(
			t_count,
			X, Y_hat, 1, output_len * t_count,
			MSE,
			&Y, host_cpp_pointer_output, gradient_hyperparameters()
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

	NN* n = NN_constructor()
		.append_layer(Dense, LSTM, 128)
		.append_layer(Dense, LSTM, 64)
		.append_layer(Dense, LSTM, 48)
		.append_layer(Dense, LSTM, 32)
		.append_layer(Dense, Neuron, 32)
		.append_layer(Dense, Neuron, out_len)
		.construct(in_len, no_optimizer, 0);


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
	hyperparameters.gradient_clip = 1;
	hyperparameters.learning_rate = .1;

	const size_t epoch_n = 5000;
	for (size_t i = 0; i < epoch_n; i++)
	{
		data_t* Y = 0;
		data_t* activations = 0;
		data_t* execution_values = 0;
		if (i % 10 == 0)
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

			if (i % 10 == 0)
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

	NN* n = NN_constructor()
		//.append_layer(NEAT, Neuron, 1, sigmoid)
		.append_layer(NEAT, Neuron, 5, sigmoid)
		.append_layer(NEAT, Neuron, output_len, activations_last_entry)
		.construct(input_len, no_optimizer);
	
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
	const size_t input_len = 2;
	const size_t output_len = 4;
	const size_t max_t_count = 20;

	NN* value_function = NN_constructor()
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 1, no_activation)
		.construct(input_len, no_optimizer);

	NN* agent = NN_constructor()
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 20)
		.append_layer(Dense, Neuron, 15)
		.append_layer(Dense, Neuron, 10)
		.append_layer(Dense, Neuron, output_len)
		.construct(input_len);

	PPO_hyperparameters parameters;
	parameters.max_training_steps = 15;
	parameters.GAE.gamma = .1;
	parameters.GAE.value_function.learning_rate = .01;
	parameters.policy.learning_rate = .001;
	parameters.clip_ratio = .1;
	parameters.max_kl_divergence_threshold = .01;

	const size_t epochs = 6000;
	for (size_t epoch_i = 0; epoch_i < epochs; epoch_i++)
	{
		data_t* initial_state = 0;
		data_t* trajectory_inputs = 0;
		data_t* trajectory_outputs = 0;

		data_t X[output_len]{};
		data_t rewards[max_t_count]{};
		data_t* Y = 0;
		data_t current_pos[2]{};
		
		data_t target_pos[2]{};
		size_t negatives = epoch_i % 4;
		target_pos[0] = (3 + rand() % 5) * (1 - 2 * (negatives == 0 || negatives == 3));
		target_pos[1] = (3 + rand() % 5) * (1 - 2 * (negatives == 1 || negatives == 3));

		size_t hit_count = 0;
		size_t t_count = 0;
		data_t mean_output[output_len]{};
		for (size_t t = 0; t < max_t_count; t++, t_count++)
		{
			X[0] = (target_pos[0] - current_pos[0]) / 8.0;
			X[1] = (target_pos[1] - current_pos[1]) / 8.0;

			Y = agent->PPO_execute(X, &initial_state, &trajectory_inputs, &trajectory_outputs, t);
			for (size_t i = 0; i < output_len; i++)
				mean_output[i] += Y[i];

			data_t movement[2]{};
			movement[0] += fmin(Y[0] * (Y[0] > Y[1]), .7);
			movement[0] -= fmin(Y[1] * (Y[1] >= Y[0]), .7);
			int i = 0;
			rewards[t] += abs(target_pos[i] - current_pos[i]) - abs(target_pos[i] - current_pos[i] + movement[i]);

			movement[1] += fmin(Y[2] * (Y[2] > Y[3]), .7);
			movement[1] -= fmin(Y[3] * (Y[3] >= Y[2]), .7);
			i = 1;
			rewards[t] += abs(target_pos[i] - current_pos[i]) -  abs(target_pos[i] - current_pos[i] + movement[i]);


			current_pos[0] += movement[0];
			current_pos[1] += movement[1];

			if (abs(target_pos[0] - current_pos[0]) < 1 && abs(target_pos[1] - current_pos[1]) < 1)
			{
				hit_count++;
				rewards[t] += 4;

				current_pos[0] = 0;
				current_pos[1] = 0;

				target_pos[0] = (3 + rand() % 5) * (1 - 2 * (rand() % 2));
				target_pos[1] = (3 + rand() % 5) * (1 - 2 * (rand() % 2));
			}

			delete[] Y;
		}
		agent->PPO_train(t_count, &initial_state, &trajectory_inputs, &trajectory_outputs, rewards, true, value_function, parameters);
		data_t mean_reward = 0;
		for (size_t i = 0; i < t_count; i++) mean_reward += rewards[i];
		mean_reward /= t_count;
		printf("i: %i hit_count: %i mean reward: %.4f target pos: %.1f %.1f mean output: ", epoch_i, hit_count, mean_reward, target_pos[0], target_pos[1]);
		for (size_t i = 0; i < output_len; i++)
			printf("%.2f ", mean_output[i] / t_count);
		printf("\n");
	}
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
	test_LSTM();
	//minimal_case();
	//test_PPO();
}
