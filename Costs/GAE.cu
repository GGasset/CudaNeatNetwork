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
	if (!discounted_rewards) throw;

	cudaMemset(discounted_rewards, 0, sizeof(data_t) * t_count);

	calculate_discounted_rewards kernel(t_count / 32 + (t_count % 32 > 0), 32) (
		t_count, parameters.gamma, rewards, discounted_rewards
	);
	cudaDeviceSynchronize();

	data_t* value_functions = 0;
	value_functions = value_function_estimator->batch_execute(
		value_function_state, t_count, device_arr
	);
	for (size_t i = 0; i < parameters.training_steps; i++)
		value_function_estimator->training_batch(
			t_count,
			value_function_state, discounted_rewards, 0, t_count,
			CostFunctions::MSE, 
			0, null_arr, parameters.value_function
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

// rewards must be of size n_executions = parallel_execution_n * t_count and write_arr must be of size n_executions * t_count
// if output is PRAM_add ed discounted rewards will remain 
__global__ void g_discounted_rewards(data_t *rewards, size_t parallel_execution_n, size_t t_count, data_t gamma, data_t *write_arr)
{
	size_t tid = get_tid();

	size_t n_executions = parallel_execution_n * t_count;
	size_t total_outs = n_executions * t_count;
	if (tid >= total_outs) return;

	size_t execution_i = tid / t_count;
	size_t t = execution_i % t_count;
	size_t addition_arr_i = tid % t_count;

	write_arr[tid] = 0;
	if (addition_arr_i > t) return;

	size_t addition_arr_len = t_count - t;

	size_t discounted_sequence_i = addition_arr_i - t;
	data_t adjusted_gamma = pow(gamma, discounted_sequence_i);

	size_t parallel_execution_line_i = execution_i / t_count;
	size_t reward_i = parallel_execution_line_i * t_count + addition_arr_i;
	write_arr[tid] = (adjusted_gamma * rewards[reward_i]) / addition_arr_len;
}

// Rewards are the appended rewards of each execution line
// Output is the discounted rewards of the t_count sequence one execution line after the other
__host__ data_t *get_discounted_rewards(size_t parallel_executions_n, size_t t_count, data_t gamma, data_t *rewards)
{
	size_t n_executions = parallel_executions_n * t_count;
	size_t expanded_n_executions = n_executions * t_count;

    data_t *unsummed_discounted_rewards = cudaCalloc<data_t>(expanded_n_executions);
	if (!unsummed_discounted_rewards) return 0;

	g_discounted_rewards n_threads(expanded_n_executions) (rewards, parallel_executions_n, t_count, gamma, unsummed_discounted_rewards);
	cudaDeviceSynchronize();

	// PRAM add
	data_t *discounted_rewards = multi_PRAM_add(unsummed_discounted_rewards, t_count, n_executions);
	cudaFree(unsummed_discounted_rewards);
	return discounted_rewards;
}

data_t *get_advantages(size_t parallel_executions_n, size_t t_count, NN *estimator, data_t *state, size_t state_len, GAE_hyperparameters gae, data_t *rewards)
{ 
	size_t n_executions = parallel_executions_n * t_count;
    size_t expected_state_len = n_executions * estimator->get_input_length();

	if (state_len != expected_state_len || !estimator || !state || !rewards) throw;
	if (estimator->get_output_length() != 1) throw;

	data_t *device_rewards = cuda_clone_arr(rewards, n_executions);
	if (!device_rewards) throw;

	data_t *discounted_rewards = get_discounted_rewards(parallel_executions_n, t_count, gae.gamma, device_rewards);
	if (!discounted_rewards) throw;
	
	cudaFree(device_rewards);


	if (gae.use_reward_normalization)
	{
		data_t *normalized_rewards = gae.reward_normalization.incoming_vals(discounted_rewards, n_executions, false, device_arr);
		cudaFree(discounted_rewards);
		discounted_rewards = normalized_rewards;
	}

	data_t *device_state = cuda_clone_arr(state, state_len);
	if (!device_state) throw;
	data_t *value_functions = 0;

	data_t *activations = 0;
	data_t *execution_values = 0;
	if (!estimator->is_recurrent())
		value_functions = estimator->execute(n_executions, 0, device_state, state_len, true, device_arr, &activations, &execution_values);
	else
	{
		throw;
		//for (size_t i = 0; )
	}
	/*if (gae.training_steps == 1)
	{
		data_t *cost_derivative = 
			MSE_derivative(n_executions, 1, activations, estimator->get_neuron_count(), 1, discounted_rewards, n_executions, false);
		data_t *grads = estimator->backpropagate(n_executions, 1, cost_derivative, n_executions, activations, execution_values, gae.value_function);
		estimator->subtract_gradients(n_executions, 1, grads, gae.value_function);

		cudaFree(cost_derivative);
		cudaFree(grads);
	}*/

	for (size_t i = 0; i < gae.training_steps; i++)
	{
		if (!activations || !execution_values) estimator->execute(n_executions, 0, device_state, state_len, true, device_arr, &activations, &execution_values);

		data_t *cost_derivative = 
			MSE_derivative(n_executions, 1, activations, estimator->get_neuron_count(), 1, discounted_rewards, n_executions, false);
		data_t *grads = estimator->backpropagate(n_executions, 1, cost_derivative, n_executions, activations, execution_values, gae.value_function);
		estimator->subtract_gradients(n_executions, 1, grads, gae.value_function);

		cudaFree(cost_derivative);
		cudaFree(grads);

		cudaFree(activations);
		activations = 0;
		cudaFree(execution_values);
		execution_values = 0;

	};

	if (!gae.use_GAE)
	{
		add_arrays n_threads(n_executions) (value_functions, discounted_rewards, value_functions, n_executions, n_executions, false, true);
		cudaDeviceSynchronize();

		//cudaFree(value_functions);
		cudaFree(discounted_rewards);
		cudaFree(device_state);

		// Value functions is being used to store advantages (deltas)
		return value_functions;
	}

	cudaFree(value_functions);
	cudaFree(discounted_rewards);
	cudaFree(device_state);

	throw;
}
