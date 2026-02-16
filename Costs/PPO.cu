
# include "PPO.cuh"

void initialize_mem(
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO::PPO_internal_memory *mem_pntr
)
{
	size_t n_env = hyperparameters.vecenvironment_count;
	mem_pntr->n_env = n_env;

	mem_pntr->initial_internal_states.resize(n_env, 0);
	mem_pntr->current_internal_states.resize(n_env, 0);

	mem_pntr->initial_value_internal_states.resize(n_env, 0);

	mem_pntr->was_memory_deleted_before.resize(n_env, std::vector<bool>());
	mem_pntr->trajectory_inputs.resize(n_env, 0);
	mem_pntr->trajectory_outputs.resize(n_env, 0);

	mem_pntr->rewards.resize(n_env, 0);
	mem_pntr->add_reward_calls_n.resize(n_env, 0);

	mem_pntr->n_env_executions.resize(n_env, 0);

	size_t value_func_state_value_count = 0;
	data_t *value_func_state = value_function->get_hidden_state(&value_func_state_value_count);
	mem_pntr->value_internal_state_length = value_func_state_value_count;

	size_t policy_state_value_count = 0;
	data_t *policy_state = policy->get_hidden_state(&policy_state_value_count);
	mem_pntr->policy_internal_state_length = policy_state_value_count;

	for (size_t i = 0; i < n_env; i++)
	{
		mem_pntr->initial_internal_states[i]
			= cuda_clone_arr(policy_state, policy_state_value_count);
		mem_pntr->current_internal_states[i]
			= cuda_clone_arr(policy_state, policy_state_value_count);

		mem_pntr->initial_value_internal_states[i]
			= cuda_clone_arr(value_func_state, value_func_state_value_count);
	}
	cudaFree(value_func_state);
	value_func_state = 0;
	cudaFree(policy_state);
	policy_state = 0;
	mem_pntr->is_initialized = true;
}

void PPO_initialization(
	data_t *X,
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO::PPO_internal_memory *mem_pntr, size_t env_i
)
{
	PPO::PPO_internal_memory mem = *mem_pntr;

	// Initialization
	if (!mem.n_env)
	{
		initialize_mem(value_function, policy, hyperparameters, mem_pntr);
		mem = *mem_pntr;
	}
	// Checks
	if (!hyperparameters.vecenvironment_count
		|| !X || !value_function || !policy
		|| value_function->get_input_length() != policy->get_input_length()
		|| mem.n_env_executions[env_i] != mem.add_reward_calls_n[env_i] // Irregular add_rewards_call
		|| !hyperparameters.steps_before_training
		|| !hyperparameters.max_training_steps
		|| !hyperparameters.mini_batch_count
	 )
		(free_PPO_data(mem_pntr), throw);
	
	if (mem.n_env != hyperparameters.vecenvironment_count) throw;
	if (mem.n_env && !mem.is_initialized)
	{ // First call after training

		mem.is_initialized = true;
	}
	*mem_pntr = mem;
}

data_t *PPO_execution(
	data_t *X, size_t env_i,
	NN *policy,
	PPO::PPO_internal_memory *mem_pntr, arr_location output_kind,
	bool delete_memory_before
)
{
	PPO::PPO_internal_memory mem = *mem_pntr;

	mem.was_memory_deleted_before[env_i].push_back(delete_memory_before);
	if (delete_memory_before)
	{
		policy->delete_memory();
	}
	else
	{
		policy->set_hidden_state(mem.current_internal_states[env_i], true);
		mem.current_internal_states[env_i] = 0;
	}

	size_t in_len = policy->get_input_length();
	size_t out_len = policy->get_output_length();
	data_t *X_tmp = 0;
	cudaMalloc(&X_tmp, sizeof(data_t) * in_len);
	cudaMemcpy(X_tmp, X, sizeof(data_t) * in_len, cudaMemcpyDefault);
	
	size_t prev_n_executions = mem.n_env_executions[env_i];
	data_t *Y = policy->inference_execute(X_tmp, device_arr);
	mem.n_env_executions[env_i]++;

	// Insert to memory
	mem.trajectory_inputs[env_i] =
		cuda_append_array(mem.trajectory_inputs[env_i], in_len * prev_n_executions,
						  X_tmp, in_len, true);
	mem.trajectory_outputs[env_i] =
		cuda_append_array(mem.trajectory_outputs[env_i], out_len * prev_n_executions,
						  Y, out_len, true);
	
	
	mem.current_internal_states[env_i] = policy->get_hidden_state();

	data_t *out = alloc_output(out_len, output_kind);
	if (out) cudaMemcpy(out, Y, sizeof(data_t) * out_len, cudaMemcpyDefault);

	cudaFree(X_tmp);
	cudaFree(Y);
	*mem_pntr = mem;
	return out;
}

void recurrent_PPO_miniBatch(
	NN *policy, PPO_hyperparameters hyperparameters, size_t minibatch_vec_start, size_t mbatch_nenvs,
	std::vector<data_t*> env_X, std::vector<data_t*> env_Y, std::vector<data_t*> advantages,
	std::vector<data_t*> policy_initial_states, std::vector<std::vector<bool>> was_mem_deleted
)
{
	size_t neuron_count = policy->get_neuron_count();
	size_t output_len = policy->get_output_length();
	size_t output_activations_start = policy->get_output_activations_start();
	size_t gradient_count = policy->get_gradient_count_per_t();
	NN     *policy_clone = policy->clone();

	size_t steps_per_env = hyperparameters.steps_before_training;
	size_t n_envs = hyperparameters.vecenvironment_count;

	if (env_X.size() != n_envs || env_Y.size() != n_envs || advantages.size() != n_envs
	|| policy_initial_states.size() != n_envs || was_mem_deleted.size() != n_envs)
		throw;

	int stop = false;
	data_t *collected_gradients = 0;
	size_t train_i;
	for (train_i = 0; train_i < hyperparameters.max_training_steps && !stop; train_i++)
	{
		data_t *minibatch_grads = 0;
		data_t total_kl_divergence = 0;
		for (size_t env_i = minibatch_vec_start; env_i - minibatch_vec_start < mbatch_nenvs; env_i++)
		{
			policy_clone->set_hidden_state(policy_initial_states[env_i], false);
			data_t* execution_values = 0;
			data_t* activations = 0;
			data_t *Y = 0;
			policy_clone->training_execute(steps_per_env,
				env_X[env_i], &Y, device_arr,
				&execution_values, &activations, 0, &was_mem_deleted[env_i]
			);
			if (!Y) throw;

			data_t *costs = 0;
			cudaMalloc(&costs, sizeof(data_t) * neuron_count * steps_per_env);
			cudaMemset(costs, 0, sizeof(data_t) * neuron_count * steps_per_env);
			total_kl_divergence += PPO_derivative(
				steps_per_env, output_len, neuron_count,
				env_Y[env_i], Y, advantages[env_i], costs,
				output_activations_start, 
				hyperparameters.clip_ratio
			);
			cudaFree(Y);

			data_t *current_grads = 0;
			policy_clone->backpropagate(
				steps_per_env, 
				costs, activations, execution_values,
				&current_grads, hyperparameters.policy
			);
			cudaFree(costs);
			cudaFree(activations);
			cudaFree(execution_values);

			collected_gradients = cuda_append_array(
				collected_gradients, gradient_count * (train_i * mbatch_nenvs * steps_per_env + env_i * steps_per_env),
				current_grads, gradient_count * steps_per_env, 
				true
			);
			minibatch_grads = cuda_append_array(
				minibatch_grads, gradient_count * env_i * steps_per_env,
				current_grads, gradient_count * steps_per_env,
				true
			);
			cudaFree(current_grads);
		}
		stop = hyperparameters.max_kl_divergence_threshold != 0 && fabs(total_kl_divergence / (steps_per_env * mbatch_nenvs)) > hyperparameters.max_kl_divergence_threshold;
		for (size_t i = 0; i < mbatch_nenvs * steps_per_env; i++)
			policy_clone->subtract_gradients(minibatch_grads, i * gradient_count, hyperparameters.policy);
		
		cudaFree(minibatch_grads);
	}
	delete policy_clone;

	for (size_t i = 0; i < train_i * mbatch_nenvs * steps_per_env; i++)
		policy->subtract_gradients(collected_gradients, i * gradient_count, hyperparameters.policy);
	
	cudaFree(collected_gradients);
}

void non_recurrent_PPO_miniBatch(
	NN *policy, PPO_hyperparameters hyperparameters,
	data_t *X, data_t *trajectory_Y, data_t *advantages, size_t data_point_count
)
{
	size_t neuron_count = policy->get_neuron_count();
	size_t output_len = policy->get_output_length();
	size_t output_activations_start = policy->get_output_activations_start();
	size_t gradient_count = policy->get_gradient_count_per_t();
	NN     *policy_clone = policy->clone();

	data_t *collected_gradients = 0;
	bool stop = false;
	size_t train_i;
	for (train_i = 0; train_i < hyperparameters.max_training_steps && !stop; train_i++)
	{
		data_t* execution_values = 0;
		data_t* activations = 0;
		data_t* Y = 0;
		policy_clone->training_execute(data_point_count, X, &Y, device_arr, &execution_values, &activations);
		if (!Y || !execution_values || !activations) throw;

		size_t costs_len = neuron_count * data_point_count;
		data_t *costs = 0;
		cudaMalloc(&costs, sizeof(data_t) * costs_len);
		cudaMemset(costs, 0, sizeof(data_t) * costs_len);

		data_t kl_divergence = PPO_derivative(
			data_point_count, output_len, neuron_count,
			trajectory_Y, Y, advantages, costs, output_activations_start,
			hyperparameters.clip_ratio
		);
#ifdef DEBUG
		if (kl_divergence > .1) printf("KL divergence too high! %.3f\n", kl_divergence);
#endif
		stop = hyperparameters.max_kl_divergence_threshold != 0 && fabs(kl_divergence) > hyperparameters.max_kl_divergence_threshold;
		cudaFree(Y);

		data_t* gradients = 0;
		policy_clone->backpropagate(
			data_point_count,
			costs, activations, execution_values, &gradients, hyperparameters.policy
		);
		cudaFree(costs);
		cudaFree(activations);
		cudaFree(execution_values);

		for (size_t t = 0; t < data_point_count; t++)
			policy_clone->subtract_gradients(gradients, gradient_count * t, hyperparameters.policy);
		collected_gradients = cuda_append_array(collected_gradients, gradient_count * data_point_count * train_i,
			gradients, gradient_count * data_point_count, true);
		cudaFree(gradients);
	}
	delete policy_clone;
	for (size_t i = 0; i < data_point_count * train_i; i++)
		policy->subtract_gradients(collected_gradients, gradient_count * i, hyperparameters.policy);
	cudaFree(collected_gradients);

}

void PPO_train(
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO::PPO_internal_memory *mem_pntr
)
{
	PPO::PPO_internal_memory mem = *mem_pntr;
	bool is_recurrent = value_function->is_recurrent() || policy->is_recurrent();

	std::vector<data_t *> advantages;
	for (size_t env_i = 0; env_i < mem.n_env; env_i++)
	{
		value_function->set_hidden_state(mem_pntr->initial_value_internal_states[env_i], false);
		assert(hyperparameters.steps_before_training == mem.add_reward_calls_n[env_i]);

		size_t advantage_buffer_len = 0;
		data_t *advantage_buffer = 0;

		size_t start_i = 0;
		for (size_t i = 1; i < hyperparameters.steps_before_training; i++)
			if (mem.was_memory_deleted_before[env_i][i] || i + 1 == hyperparameters.steps_before_training)
			{
				size_t steps_to_calculate = i - start_i;
				data_t *current_advantages = calculate_advantage(
					steps_to_calculate,
					value_function, mem.trajectory_inputs[env_i] + (start_i * value_function->get_input_length()),
					hyperparameters.GAE, false, false,
					mem.rewards[env_i] + start_i, false, false
				);
				advantage_buffer = cuda_append_array(
					advantage_buffer, advantage_buffer_len,
					current_advantages, steps_to_calculate,
					true
				);
				advantage_buffer_len += steps_to_calculate;

				start_i = i;
			}

		advantages.push_back(advantage_buffer);
		mem.initial_value_internal_states[env_i] = value_function->get_hidden_state();
	}

	size_t steps_per_env = hyperparameters.steps_before_training;
	size_t env_n = hyperparameters.vecenvironment_count;
	size_t total_execution_count = env_n * steps_per_env;
	size_t minibatch_count = hyperparameters.mini_batch_count;
	if (is_recurrent)
	{
		size_t envs_per_minib = env_n / minibatch_count;
		size_t last_envs_per_minib = env_n % minibatch_count;
		std::vector<std::tuple<size_t, size_t>> minib_start_n_size;
		for (size_t i = 0; i < minibatch_count; i++)
			minib_start_n_size.push_back(
				{
					i * envs_per_minib,
					envs_per_minib * (i < minibatch_count - 1) + last_envs_per_minib * (i == minibatch_count - 1)
				}
			);
		
		// shuffle vector
		vec_shuffle_inplace<std::tuple<size_t, size_t>>(minib_start_n_size);

		for (size_t i = 0; i < minibatch_count; i++)
			recurrent_PPO_miniBatch(
				policy, hyperparameters,
				std::get<0>(minib_start_n_size[i]), std::get<1>(minib_start_n_size[i]),
				mem.trajectory_inputs, mem.trajectory_outputs, advantages,
				mem.initial_internal_states, mem.was_memory_deleted_before
			);
		
	}
	else
	{
		// Prepare arrays

		data_t *appended_X = 0;
		data_t *appended_Y = 0;
		data_t *appended_advantages = 0;
		for (size_t i = 0; i < hyperparameters.vecenvironment_count; i++)
		{
			appended_X = cuda_append_array(
				appended_X, policy->get_input_length() * steps_per_env * i,
				mem.trajectory_inputs[i], policy->get_input_length() * steps_per_env,
				true
			);
			appended_Y = cuda_append_array(
				appended_Y, policy->get_output_length() * steps_per_env * i,
				mem.trajectory_outputs[i], policy->get_output_length() * steps_per_env,
				true
			);
			appended_advantages = cuda_append_array(
				appended_advantages, steps_per_env * i,
				advantages[i], steps_per_env,
				true
			);

		}

		// add shuffling
		size_t *shuffling_keys = std::get<0>(cud_get_shuffled_indices(total_execution_count));
		cuda_sort_by_key(&appended_X, shuffling_keys, total_execution_count);
		cuda_sort_by_key(&appended_Y, shuffling_keys, total_execution_count);
		cuda_sort_by_key(&appended_advantages, shuffling_keys, total_execution_count);

		// Create training data

		size_t items_per_minib = steps_per_env / hyperparameters.mini_batch_count;
		size_t last_item_count = items_per_minib + steps_per_env % hyperparameters.mini_batch_count;
		for (size_t i = 0; i < hyperparameters.mini_batch_count; i++)
		{
			data_t *X_pntr = appended_X + i * items_per_minib * policy->get_input_length();
			data_t *Y_pntr = appended_Y + i * items_per_minib * policy->get_output_length();
			data_t *advantage_pntr = appended_advantages + i * items_per_minib;
			size_t value_count = 
				i < hyperparameters.mini_batch_count - 1 ? items_per_minib : last_item_count;

			non_recurrent_PPO_miniBatch(
				policy, hyperparameters, X_pntr, Y_pntr, advantage_pntr, value_count
			);
		}
		cudaFree(appended_X);
		cudaFree(appended_Y);
		cudaFree(appended_advantages);
	}
	for (size_t i = 0; i < advantages.size(); i++)
		cudaFree(advantages[i]);
	*mem_pntr = mem;
}

void PPO_data_cleanup(PPO::PPO_internal_memory *mem_pntr)
{
	PPO::PPO_internal_memory mem = *mem_pntr;

	for (size_t i = 0; i < mem.n_env; i++)
	{
		cudaFree(mem.initial_value_internal_states[i]);
		mem.initial_value_internal_states[i] = 0;

		cudaFree(mem.initial_internal_states[i]);
		mem.initial_internal_states[i] = 0;

		mem.initial_internal_states[i] 
			= cuda_clone_arr(mem.current_internal_states[i], mem.policy_internal_state_length);

		mem.was_memory_deleted_before[i] = std::vector<bool>();

		cudaFree(mem.trajectory_inputs[i]);
		mem.trajectory_inputs[i] = 0;

		cudaFree(mem.trajectory_outputs[i]);
		mem.trajectory_outputs[i] = 0;

		cudaFree(mem.rewards[i]);
		mem.rewards[i] = 0;

		mem.add_reward_calls_n[i] = 0;

		mem.n_env_executions[i] = 0;

		mem.is_initialized = 0;
	}
	*mem_pntr = mem;
}

data_t *PPO::PPO_execute_train(
	data_t *X,  size_t env_i,
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO::PPO_internal_memory *mem_pntr, arr_location output_kind,
	bool delete_memory_before
)
{
	PPO::PPO_internal_memory mem = *mem_pntr;
	PPO_initialization(
		X, value_function, policy,
		hyperparameters, &mem,
		env_i
	);
	*mem_pntr = mem;

	// Execution
	data_t *out = PPO_execution(
		X, env_i, policy, &mem, output_kind, delete_memory_before
	);
	*mem_pntr = mem;
	
	return out;
}

void PPO::add_reward(
	data_t reward, size_t env_i, PPO_internal_memory *mem_pntr,
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters
)
{
	mem_pntr->rewards[env_i] = cuda_push_back(mem_pntr->rewards[env_i], mem_pntr->add_reward_calls_n[env_i], reward, true);
	mem_pntr->add_reward_calls_n[env_i]++;

	// Training Checks
	PPO::PPO_internal_memory mem = *mem_pntr;
	bool start_training = 1;
	for (size_t i = 0; i < mem.n_env && start_training; i++)
	{
		if (mem.n_env_executions[i] > hyperparameters.steps_before_training)
		{
			free_PPO_data(mem_pntr);
			// all vecenvironment execute should be called homogeneously
			throw "Irregular env_steps not Implemented";	
		}
		start_training = start_training 
			&& mem.add_reward_calls_n[i] == hyperparameters.steps_before_training
			&& mem.n_env_executions[i] == hyperparameters.steps_before_training;
	}
	*mem_pntr = mem;
	if (!start_training) return;
	
#ifdef DEBUG
	std::cout << "Starting training" << std::endl;
#endif

	// Training loop
	PPO_train(value_function, policy, hyperparameters, &mem);
	
	// Mem cleanup (sets initial states to current states, does not free current states)	
	*mem_pntr = mem;
	PPO_data_cleanup(mem_pntr);
}

bool PPO::free_PPO_data(PPO::PPO_internal_memory *mem)
{
	PPO_data_cleanup(mem);

	for (size_t i = 0; i < mem->n_env; i++)
	{
		cudaFree(mem->initial_internal_states[i]);
		mem->initial_internal_states[i] = 0;

		cudaFree(mem->current_internal_states[i]);
		mem->current_internal_states[i] = 0;

		cudaFree(mem->initial_value_internal_states[i]);
		mem->initial_value_internal_states[i] = 0;
	}
	
	mem->n_env = 0;

	return 0;
}
