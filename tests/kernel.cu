
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

#include "NN_constructor.h"

#include "PPO.cuh"
#include "test_env.h"

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
		.append_layer(Dense, Neuron, output_len, _tanh)
		.append_layer(Dense, Neuron, output_len, _tanh)
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
	hyperparameters.learning_rate = .01;

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

// Bugs: saving loading, global gradient clip, check kl divergence
// TODO: move value function training loop outside of advantage calculation for greater efficiency
// TODO: set biases to 0, then orthogonal in initialization, create modular initializer
static void test_PPO(int argc)
{
	const size_t n_envs = 32;
	test_env env = test_env(n_envs);

	size_t input_len = env.get_observation_count();
	size_t output_len = env.get_action_count();

	optimizer_hyperparameters value_function_optimizer;
	value_function_optimizer.adam.epsilon = 1e-5;
	NN* value_function; 
	if (argc == 1)
		value_function = NN_constructor()
		.append_layer(Dense, Neuron, 128)
		.append_layer(Dense, Neuron, 80)
		.append_layer(Dense, Neuron, 50)
		.append_layer(Dense, Neuron, 50)
		.append_layer(Dense, Neuron, 20, _tanh)
		.append_layer(Dense, Neuron, 1, _tanh)
		.construct(input_len, value_function_optimizer);
	else
	{
		value_function = NN::load("./network_value_function_test.n");
		if (!value_function)
		{
			std::cerr << "could not load value function network" << std::endl;
			throw;
		}
	}

	optimizer_hyperparameters policy_optimizer;
	policy_optimizer.adam.epsilon = 1e-5;
	NN* policy;
	
	if (argc == 1)
		policy = NN_constructor()
		.append_layer(Dense, Neuron, 128)
		.append_layer(Dense, Neuron, 80)
		.append_layer(Dense, Neuron, 50)
		.append_layer(Dense, Neuron, 50)
		.append_layer(Dense, Neuron, 20, _tanh)
		.append_layer(Dense, Neuron, output_len, softmax)
		.construct(input_len, policy_optimizer);
	else
	{
		policy = NN::load("./network_policy_test.n");
		if (!policy)
		{
			std::cerr << "could not load policy network" << std::endl;
			throw;
		}
	}



	const double learning_rate_anhealing_coeff = 1 - 1e-4;
	PPO_hyperparameters parameters;

	parameters.GAE.use_GAE = true;
	parameters.GAE.training_steps = 3;
	parameters.GAE.gamma = .9;
	parameters.GAE.value_function.learning_rate = 1e-4;
	parameters.GAE.value_function.gradient_clip = .5;
	parameters.GAE.value_function.global_gradient_clip = .0;

	parameters.policy.gradient_clip = .5;
	parameters.policy.learning_rate = 1e-4;
	parameters.policy.global_gradient_clip = .0;

	parameters.max_training_steps = 15;
	parameters.clip_ratio = .1;
	parameters.max_kl_divergence_threshold = .0;

	parameters.policy.regularization.entropy_bonus.active = true;
	parameters.policy.regularization.entropy_bonus.entropy_coefficient = 1E-3;

	parameters.vecenvironment_count = n_envs;
	parameters.steps_before_training = 10;
	parameters.mini_batch_count = 2;

	std::vector<bool> shall_delete_memory;
	shall_delete_memory.resize(n_envs, false);
	PPO::PPO_internal_memory mem{};
	size_t exec_n = 0;
	int total_hit_count = 0;
	for (size_t i = 0; true; i++)
	{
		if ((i + 1) % 100 == 0)
		{
			value_function->save("./network_value_function_test.n");
			policy->save("./network_policy_test.n");
		}

		int hit_count = 0;
		data_t mean_output = 0;
		data_t mean_reward = 0;
		for (size_t env_i = 0; env_i < n_envs; env_i++, exec_n++)
		{
			std::vector<data_t> obs = env.get_observations(env_i);
			data_t *actions = PPO::PPO_execute_train(
				obs.data(), env_i,
				value_function, policy, parameters,
				&mem, host_cpp_pointer_output,
				shall_delete_memory[env_i]
			);
			auto [reward, end_of_episode] = env.step(actions, env_i);
			hit_count += reward == 1;
			hit_count -= reward == -1;
			//for (size_t i = 0; i < policy->get_output_length(); i++) mean_output += actions[i];
			delete[] actions;
			shall_delete_memory[env_i] = end_of_episode;

			PPO::add_reward(
				reward, env_i, &mem,
				value_function, policy,
				parameters
			);

			mean_reward += reward;
		}
		mean_reward /= n_envs;
		//mean_output /= policy->get_output_length() * n_envs;
		total_hit_count += hit_count;

		std::stringstream ss;
		ss << hit_count << ", " << env.get_mean_episode_len() << ", " << env.get_last_episode_lens_len() << ", " << mean_reward << ", " << i << ", " << exec_n << std::endl;
		std::cout << ss.str();
	}
}

int main(int argc)
{
#ifdef DETERMINISTIC
	srand(13);
#else
	srand(get_arbitrary_number());
#endif


	//cudaSetDevice(0);
	//bug_hunting();
	//test_LSTM();
	test_PPO(argc);

	printf("Last error peek: %i\n", cudaPeekAtLastError());
}
