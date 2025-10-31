#include "GAE.cuh"

__global__ void calculate_discounted_rewards(
	size_t t_count,
	data_t gamma,
	data_t *rewards,
	data_t *discounted_rewards
)
{
	size_t tid = get_tid();
	if (tid >= t_count)
		return;
	data_t discount_factor = 1;
	discounted_rewards[tid] = 0;
	for (size_t i = tid; i < t_count; i++, discount_factor *= gamma)
		discounted_rewards[tid] += rewards[i] * discount_factor;
}

__global__ void calculate_deltas(
	size_t t_count,
	data_t gamma,
	data_t *rewards,
	data_t *value_functions,
	data_t *deltas
)
{
	size_t tid = get_tid();
	if (tid >= t_count)
		return;
	deltas[tid] = -value_functions[tid] + rewards[tid];
	if (tid >= t_count - 1)
		return;
	deltas[tid] += gamma * value_functions[tid + 1];
}

__global__ void parallel_calculate_GAE_advantage(
	size_t t_count,
	data_t gamma,
	data_t lambda,
	data_t *deltas,
	data_t *advantages
)
{
	size_t tid = get_tid();
	if (tid >= t_count)
		return;

	data_t gamma_lambda = gamma * lambda;
	data_t GAE_discount = 1;
	advantages[tid] = 0;
	for (size_t i = tid; i < t_count; i++, GAE_discount *= gamma_lambda)
		advantages[tid] += GAE_discount * deltas[i];
}

data_t *calculate_advantage(
	size_t t_count,
	NN *value_function_estimator, data_t *value_function_state, GAE_hyperparameters parameters, bool is_state_on_host, bool free_state,
	data_t *rewards, bool is_reward_on_host, bool free_rewards
)
{
	if (!value_function_estimator) return (0);

	if (is_reward_on_host)
	{
		data_t* device_rewards = 0;
		cudaMalloc(&device_rewards, sizeof(data_t) * t_count * value_function_estimator->get_output_length());
		cudaMemcpy(device_rewards, rewards, sizeof(data_t) * t_count * value_function_estimator->get_output_length(), cudaMemcpyHostToDevice);
		if (free_rewards) delete[] rewards;
		rewards = device_rewards;
	}

	data_t *discounted_rewards = 0;
	cudaMalloc(&discounted_rewards, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();
	if (!discounted_rewards) throw;

	cudaMemset(discounted_rewards, 0, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();

	calculate_discounted_rewards kernel(t_count / 32 + (t_count % 32 > 0), 32) (
		t_count, parameters.gamma, rewards, discounted_rewards
	);
	cudaDeviceSynchronize();

	data_t* value_functions = 0;
	value_functions = value_function_estimator->batch_execute(
		value_function_state, t_count, cuda_pointer_output
	);
	for (size_t i = 0; i < parameters.training_steps; i++)
		value_function_estimator->training_batch(
			t_count,
			value_function_state, discounted_rewards, 0, t_count,
			CostFunctions::MSE, 
			0, no_output, parameters.value_function
		);
	
	if (!parameters.use_GAE)
	{
		data_t *advantages = 0;
		cudaMalloc(&advantages, sizeof(data_t) * t_count);
		if (!advantages) throw;
		cudaMemset(advantages, 0, sizeof(data_t) * t_count);

		add_arrays n_threads(t_count) (
			advantages, discounted_rewards, value_functions,
			t_count, t_count, false, true
		);
		cudaDeviceSynchronize();

		if (is_reward_on_host) cudaFree(rewards);
		else if (free_rewards) delete[] rewards;
		cudaFree(discounted_rewards);
		cudaFree(value_functions);
		return advantages;
	}

	data_t *deltas = 0;
	cudaMalloc(&deltas, sizeof(data_t) * t_count);
	if (!deltas) throw;
	cudaMemset(deltas, 0, sizeof(data_t) * t_count);
	
	calculate_deltas kernel(t_count / 32 + (t_count % 32 > 0), 32) (
		t_count, 
		parameters.gamma, 
		rewards, 
		value_functions, 
		deltas
	);
	cudaDeviceSynchronize();

	data_t* advantages = 0;
	cudaMalloc(&advantages, sizeof(data_t) * t_count);
	if (!advantages) throw;
	cudaMemset(advantages, 0, sizeof(data_t) * t_count); 

	parallel_calculate_GAE_advantage kernel(t_count / 32 + (t_count % 32 > 0), 32) (
		t_count,
		parameters.gamma, parameters.lambda,
		deltas, advantages
	);
	cudaDeviceSynchronize();

#ifdef DEBUG
	if (0)
	{
		printf("Discounted rewards: ");
		print_array(discounted_rewards, t_count);

		printf("Value functions: ");
		print_array(value_functions, t_count);

		printf("Deltas: ");
		print_array(deltas, t_count);

		printf("Advantages: ");
		print_array(advantages, t_count);
	}

#endif // DEBUG

	if (is_reward_on_host) cudaFree(rewards);
	else if (free_rewards) delete[] rewards;
	cudaFree(discounted_rewards);
	cudaFree(deltas);
	cudaFree(value_functions);

	return advantages;
}
