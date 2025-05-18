
#include "math.h"
#include "GAE.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <sstream>

#include "NN_constructor.h"

/*
static int increment_i(size_t size, int to_increment, size_t i)
{
	i += to_increment;+
	i -= (i - i % size) * (i >= size);
	i = (i % size) * (i < 0) + i * (i >= 0);
	return i;
}*/

template<typename t>
t min2(t a, t b)
{
	return a * (a <= b) + b * (b < a);
}

template<typename t>
t abs(t a)
{
	return a * (-1 + 2 * (a > 0));
}

// Penalizes for output close to 0 (-1) 0.5 (.875) and 1 (-1) and rewards for being close to .25 (1) and .75 (1)
// rewards up to 1 / output_count and down to -1 / output_count
void output_stabilizer(data_t output, data_t *current_reward, size_t output_count)
{
	data_t x_multiplier = 3.2;
	data_t x_shifter = 1.6;
	data_t upwards = (x_multiplier * output - x_shifter) * (-x_multiplier * output + x_shifter) + 1;
	data_t downwards = (x_multiplier * output - x_shifter) * (x_multiplier * output - x_shifter);

	data_t middle_point_y = .2;//.875;
	data_t output_multiplier = .5;
	data_t output_reward = upwards * downwards + middle_point_y;

	*current_reward += output_reward / output_count;
}

void GridTravellerPrototype()
{
	srand(101);
	cudaSetDevice(0);

	const size_t input_length = 2;
	const size_t output_length = 2;
	/*const size_t t_count = 5;
	data_t X[input_length * t_count]{};
	data_t Y_hat[output_length * t_count]{};

	for (size_t t = 0; t < t_count; t++)
	{
		for (size_t i = 0; i < input_length; i++)
		{
			X[t * input_length + i] = (.2);//  / t_count * ((t + 1)) + .2 / t_count;
		}
		 for (size_t i = 0; i < output_length; i++)
		{
			Y_hat[t * output_length + i] = .05;//(.2) / t_count * (t + 1) + .2 / t_count + (.2) / t_count * (i + 1) / output_length;
		}
	}*/


	
	NN *n = NN_constructor()
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 30, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 15, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 7, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 5, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, output_length, ActivationFunctions::sigmoid)
		.construct(input_length);

	NN *value_function = NN_constructor()
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 30, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 15, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 7, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 1, ActivationFunctions::sigmoid)
		.construct(input_length);
	
	/*NN* n = NN_constructor()
		.append_layer(ConnectionTypes::NEAT, NeuronTypes::Neuron, output_length, ActivationFunctions::sigmoid)
		.construct(input_length);*/
	
	n->stateful = true;

	data_t total_mean_r = 0;
	data_t max_total_mean_r = -100000;
	size_t total_mean_r_count = 0;

	const size_t epochs = 100000;
	const size_t max_steps = 50;
	for (size_t i = 0; i < epochs; i++)
	{
		// RL demonstration
		data_t x = 0;
		data_t y = 0;

		data_t target_x = 1;
		data_t target_y = 1;
		//target_x = target_x;
		//target_y = target_y;
		//target_x *= 1 - 2 * (data_t)(rand() % 2);
		//target_y *= 1 - 2 * (data_t)(rand() % 2);
		target_x *= 1 - 2 * (data_t)(i % 2);
		target_y *= 1 - 2 * (data_t)(i % 2);
		
		data_t initial_target_x = target_x;
		data_t initial_target_y = target_y;

		data_t inputs[input_length * max_steps];
		data_t X[input_length];
		data_t *Y = 0;
		data_t* execution_values = 0;
		data_t* activations = 0;
		
		data_t rewards[max_steps];
		data_t supervised_outputs[max_steps * 2];
		
		size_t hit_count = 0;
		bool success = false;


		size_t actual_steps = 0;
		data_t max_reward = 0;
		data_t mean_reward = 0;
		data_t mean_output[output_length] {0, 0};
		for (size_t step_i = 0; step_i < max_steps; step_i++)
		{
			if (success)
			{
				hit_count++;

				x = 0;
				y = 0;
				target_x = 0;
				target_y = 0;
				
				target_x = (rand() % hit_count);
				target_y = (rand() % hit_count);
				target_x++;
				target_y++;

				size_t negative = rand() % 2;
				target_x *= 1 - 2 * (data_t)(negative);
				target_y *= 1 - 2 * (data_t)(negative);
				
				success = false;
			}

			actual_steps++;
			rewards[step_i] = 0;

			data_t target_direction_x = target_x - x;
			data_t target_direction_y = target_y - y;
			data_t target_distance = abs(target_direction_x) + abs(target_direction_y);

			
			inputs[step_i * 2] = target_direction_x / initial_target_x;
			inputs[step_i * 2 + 1] = target_direction_y / initial_target_y;
			X[0] = (target_direction_x > 0 ? 1.5 : -1.5) * 1;
			X[1] = (target_direction_y > 0 ? 1.5 : -1.5) * 1;

			supervised_outputs[step_i * 2] = target_direction_x > 0 ? .75 : .25;
			supervised_outputs[step_i * 2 + 1] = target_direction_y > 0 ? .75 : .25;
			
			n->training_execute(
				1, X, &Y, true,
				&execution_values, &activations,
				step_i
			);

			//x += (Y[0] - .5) * 2 * 3;
			//y += (Y[1] - .5) * 2 * 3;
			x += (Y[0] > .5) ? 1 : -1;
			y += (Y[1] > .5) ? 1 : -1;

			mean_output[0] += Y[0];
			mean_output[1] += Y[1];


			data_t new_target_direction_x = target_x - x;
			data_t new_target_direction_y = target_y - y;
			data_t new_target_distance = abs(new_target_direction_x) + abs(new_target_direction_y);

			//rewards[step_i]--;
			rewards[step_i] += 
				((abs(target_direction_x) > abs(new_target_direction_x))
				+ (abs(target_direction_y) > abs(new_target_direction_y)))
				* .5;
			
			rewards[step_i] -= rewards[step_i] == 0;

			output_stabilizer(Y[0], rewards + step_i, output_length);
			output_stabilizer(Y[1], rewards + step_i, output_length);

			max_reward += (abs(rewards[step_i]) - max_reward) * (abs(rewards[step_i]) > max_reward);
			mean_reward += rewards[step_i];

			if ((success = abs(new_target_distance) < 1))
			{
				rewards[step_i] += 3;
			}
		}
		
		mean_output[0] /= actual_steps;
		mean_output[1] /= actual_steps;
		
		mean_reward /= actual_steps;

		data_t discounted_rewards[max_steps];
		const data_t gamma = .995;
		for (size_t j = 0; j < actual_steps; j++)
		{
			discounted_rewards[j] = 0;
			data_t discount_factor = 1;
			
			for (size_t k = j; k < actual_steps; k++, discount_factor *= gamma)
			{
				discounted_rewards[j] += rewards[k] * discount_factor;
			}
		}
		data_t *value_functions;
		value_function->training_batch(
			actual_steps,
			inputs,
			discounted_rewards, 1, actual_steps,
			CostFunctions::MSE,
			&value_functions, 1, gradient_hyperparameters()
		);

		// Denoted as greek lowercase delta
		data_t deltas[max_steps];
		for (size_t j = 0; j < actual_steps; j++)
		{
			data_t discount_factor = gamma;
			deltas[j] = -value_functions[j] + rewards[j] + discount_factor * (j == actual_steps - 1 ? 0 : value_functions[j + 1]);
		}
		delete[] value_functions;

		const data_t lambda = .98;
		data_t gamma_lambda = gamma * lambda;
		data_t advantages[max_steps];
		for (size_t j = 0; j < actual_steps; j++)
		{
			advantages[j] = 0;
			data_t GAE_discount = 1;
			for (size_t k = j; k < actual_steps && (/*Handle lambda=0 for performace*/1); k++, GAE_discount *= gamma_lambda)
			{
				advantages[j] += GAE_discount * deltas[k];
			}
		}
		

		n->train(actual_steps,
			execution_values, activations, 
			advantages, true, actual_steps,
			CostFunctions::log_likelyhood, gradient_hyperparameters()
		);

		/*n->train(actual_steps,
			execution_values, activations, 
			supervised_outputs, true, actual_steps * output_length,
			CostFunctions::MSE, .001, 100, .2
		);*/



		printf("     ");
		//printf("Mean reward: %.2f | final distance: %.2f | inital distance: %.2f || ", mean_reward, (abs(target_x - x) + abs(target_y - y)), abs(target_x) + abs(target_y));
		int reward_pos = (int)((2 + mean_reward) * 4);
		for (size_t j = 0; j <= 20; j++) reward_pos == j ? printf("#") : printf("-");
		
		total_mean_r += mean_reward;
		total_mean_r_count++;

		printf("  Hit count: %i | Current mean reward: %.7f | Mean Reward: %.2f | target_x: %.0f, target_y: %.0f | %i | %.3f", hit_count, mean_reward, total_mean_r / total_mean_r_count, target_x, target_y, i, max_total_mean_r);
		printf(" | Mean x output: %.2f | Mean y output %.2f      \r", mean_output[0], mean_output[1]);

		if (i % 1000 == 0)
		{
			max_total_mean_r += ((total_mean_r / total_mean_r_count) - max_total_mean_r) * (total_mean_r > max_total_mean_r);
			total_mean_r = total_mean_r_count = 0;
		}
		delete[] Y;
	}
	delete n;
	delete value_function;
	cudaDeviceReset();
	//n.deallocate();
}


void test_LSTM_cells_for_rythm_prediction()
{
	const bool stateful = 1;
	const size_t epoch_n = 10000;
	const size_t t_count = 20;
	
	const size_t input_length = 1;
	const size_t output_length = 1;

	NN* n = NN_constructor()
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 256, sigmoid)
		//.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 512, sigmoid)
		//.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 256, sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 256, sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, output_length, sigmoid)
		.construct(input_length, no_optimizer, stateful);

	/*
	NN *n = NN_constructor()
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 4)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 8)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 4)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 2)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, output_length)
	.construct(input_length);
	*/
	/*NN* n = NN_constructor()
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 30, ActivationFunctions::sigmoid)
	//.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 15, ActivationFunctions::sigmoid)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 20, ActivationFunctions::sigmoid)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 30, ActivationFunctions::sigmoid)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 20, ActivationFunctions::sigmoid)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 10, ActivationFunctions::sigmoid)
	//.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 20, ActivationFunctions::sigmoid)
	//.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 10, ActivationFunctions::sigmoid)
	.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, output_length, ActivationFunctions::sigmoid)
	.construct(input_length, stateful);*/


	data_t X[t_count * input_length];
	data_t Y_hat[t_count * output_length];

	for (size_t i = 0; i < t_count * output_length; i++)
	{
		data_t epoch_percentage = i / (float)(t_count * output_length);
		/*epoch_percentage -= (long)epoch_percentage;
		epoch_percentage *= t_count;
		while (epoch_percentage > t_count / 5.0)
			epoch_percentage -= t_count / 5.0;
		epoch_percentage /= t_count / 5.0;
		Y_hat[i] = (epoch_percentage * .7 + .15);*/
		//Y_hat[i] = sinf(epoch_percentage * 5) / 8 + .15 + .55;
		//Y_hat[i] = sinf(epoch_percentage * 3);
		//Y_hat[i] = epoch_percentage;
		//Y_hat[i] = .7;
		Y_hat[i] = i % 2 ? -.5 : .5;
		printf("%.2f ", Y_hat[i]);
	}

	printf("\n\n");
	for (size_t i = 0; i < t_count * input_length; i++)
	{
		data_t epoch_percentage = i / (float)(t_count * input_length);
		//epoch_percentage -= (long)epoch_percentage;

		//X[i] = ((epoch_percentage + .01) * (i % 2)) / 2 + .25; //* 2 - 1 + (int)(epoch_percentage * 10) % 2;
		//X[i] = !(i / input_length) + .0 - (i / input_length);
		//X[i] = Y_hat[i];
		//X[i] = (i / input_length) / (float)t_count + .05;
		//X[i] = sinf(epoch_percentage);
		//if (i % t_count == 0 && i && 1) X[i] = -1;
		if ((i % 2) && 1) X[i] = -1; else X[i] = 1;

		printf("%.2f ", X[i]);
		//X[i] -= !i;
	}

	std::string a;
	std::cin >> a;

	data_t learning_rate = .01;
	data_t costs[epoch_n];
	for (size_t i = 0; i < epoch_n; i++)
	{
		data_t *Y = 0;
		if (1)
			costs[i] = n->training_batch(
				t_count,
				X, Y_hat, true, output_length * t_count,
				CostFunctions::MSE, &Y, true, gradient_hyperparameters()
			);
		else
		{
			costs[i] = 0;
			Y = n->batch_execute(X, t_count);
		}
		if (i % 10 == 0)
			std::cout << i << " " << costs[i] << std::endl;
		for (size_t j = 0; j < output_length * t_count; j++) printf("%.2f ", Y[j]); printf("\n\n");
		//std::cout << std::endl << "----" << std::endl;
		size_t x = 10;
		for (size_t y = x + 1; y >= 1 && 0 /*&& !(epoch_n - i - 1)*/; y--)
		{
		    for (size_t x = 0; x < t_count * output_length; x++)
		    {
				int condition = ((int)(Y[x] * x)) == y - 1;
				//if (!(x % output_length))			std::cout << "|";
				if (condition) 						std::cout << "#";
				else if (Y[x] != Y[x] && !(y - 1))	std::cout << "/";
				else 								std::cout << " ";
		    }
			//std::cout << std::endl << "---" << std::endl;
			std::cout << std::endl;
		}
		delete[] Y;
		//n->delete_memory();
	}

	delete n;
	//return ;

	FILE *log_file = fopen("E:\\Code\\sine_text.csv", "wb");
	char headers[] = "cost\n";
	fwrite(headers, 1, strlen(headers), log_file);
	for (size_t i = 0; i < epoch_n; i++)
	{
		std::stringstream sline("");
		//sline << i << ", ";
		sline << costs[i] << "\n";
		std::string line = sline.str();
		fwrite(line.data(), 1, line.length(), log_file);
	}
	fclose(log_file);
}

void bug_hunting()
{
	const size_t input_len = 1;
	const size_t output_len = 2;

	const bool stateful = false;
	NN *n = NN_constructor()
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 20, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 10, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, output_len, ActivationFunctions::sigmoid)
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
			CostFunctions::MSE,
			&Y, 1, gradient_hyperparameters()
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
void test_LSTM()
{
	const data_t learning_rate = .1;
	const data_t dropout_rate = 0.2;

	const size_t in_len = 1;
	const size_t out_len = 2;

	NN* n = NN_constructor()
		.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 128, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 64)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 48)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::LSTM, 32)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 32, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, out_len, ActivationFunctions::sigmoid)
		.construct(in_len, no_optimizer, 0);


	const size_t t_count = 3;
	data_t X[in_len * t_count * 2] {};
	data_t Y_hat[out_len * t_count * 2] {};

	for (size_t i = 0; i < in_len * t_count; i++)
		X[i] = !i ? -.5 : 0;
	for (size_t i = 0; i < in_len * t_count; i++)
		X[out_len * t_count + i] = !i ? .5 : 0;

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
				X + in_len * t_count * j, &Y, true,
				&execution_values, &activations
			);
			data_t cost = n->train(
				t_count,
				execution_values, activations, Y_hat + out_len * t_count * j, true, out_len * t_count,
				CostFunctions::MSE, gradient_hyperparameters()
			);

			if (i % 10 == 0)
				printf("%i | %.4f | %.4f, %.4f\n", i, 0, Y[out_len * t_count - 2], Y[out_len * t_count - 1]);

			delete[] Y;
		}
	}
}

void minimal_case()
{
	const size_t input_len = 1;
	const size_t output_len = 1;
	const size_t t_count = 1;

	data_t X[input_len * t_count]{};
	for (size_t i = 0; i < input_len * t_count; i++)
	{
		X[i] = .5;
	}

	data_t Y_hat[output_len * t_count]{};
	for (size_t i = 0; i < output_len * t_count; i++)
	{
		Y_hat[i] = .5;
	}

	NN* n = NN_constructor()
		.append_layer(Dense, Neuron, 5, sigmoid)
		.append_layer(Dense, Neuron, output_len, sigmoid)
		.construct(input_len, Adam);

	gradient_hyperparameters hyperparameters;
	size_t epochs = 1000;
	for (size_t epoch = 0; epoch < epochs; epoch++)
	{
		data_t *Y = 0;
		n->training_batch(t_count, X, Y_hat, true, output_len * t_count, MSE,
			&Y, true, hyperparameters);
		printf("\n");
		for (size_t i = 0; i < output_len * t_count; i++)
			printf("%.2f ", Y[i]);
		printf("\n\n");
		delete[] Y;
	}
}

int main()
{
	//cudaSetDevice(0);
	//GridTravellerPrototype();
	//test_LSTM_cells_for_rythm_prediction();
	bug_hunting();
	//test_LSTM();
	//minimal_case();
}
