
# include "PPO.cuh"

void initialize_mem(
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO_internal_memory *mem_pntr
)
{
	size_t n_env = hyperparameters.vecenvironment_count;
	mem_pntr->n_env = n_env;

	mem_pntr->initial_internal_states.resize(n_env, 0);
	mem_pntr->current_internal_states.resize(n_env, 0);

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
		mem_pntr->current_value_internal_states[i]
			= cuda_clone_arr(value_func_state, value_func_state_value_count);
	}
	cudaFree(value_func_state);
	value_func_state = 0;
	cudaFree(policy_state);
	policy_state = 0;
	

	mem_pntr->is_initialized = 1;
}

void PPO_initialization(
	data_t *X,
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO_internal_memory *mem_pntr, size_t env_i
)
{
	// Checks
	PPO_internal_memory mem = *mem_pntr;
	if (!hyperparameters.vecenvironment_count
		|| !X || !value_function || !policy
		|| value_function->get_input_length() != policy->get_input_length()
		|| mem.n_env_executions[env_i] != mem.add_reward_calls_n[env_i] // Irregular add_rewards_call
		|| !hyperparameters.steps_before_training
		|| !hyperparameters.max_training_steps
		|| !hyperparameters.mini_batch_count
	 )
		(free_PPO_data(mem_pntr), throw);
	
	// Initialization
	if (!mem.n_env)
	{
		initialize_mem(value_function, policy, hyperparameters, mem_pntr);
		mem = *mem_pntr;
	}
	if (mem.n_env != hyperparameters.vecenvironment_count) throw;
	if (!mem.is_initialized) // states should be initialized and n_env, everything else should not
	{

	}
	*mem_pntr = mem;
}

data_t *PPO_execution(
	data_t *X, size_t env_i,
	NN *value_function, NN *policy,
	PPO_internal_memory *mem_pntr, output_pointer_type output_kind,
	bool delete_memory_before
)
{
	PPO_internal_memory mem = *mem_pntr;

	mem.was_memory_deleted_before[env_i].push_back(delete_memory_before);
	if (delete_memory_before)
	{
		value_function->delete_memory();
		policy->delete_memory();
	}
	else
	{
		value_function->set_hidden_state(mem.current_value_internal_states[env_i], true);
		policy->set_hidden_state(mem.current_internal_states[env_i], true);
	}

	size_t in_len = policy->get_input_length();
	size_t out_len = policy->get_output_length();
	data_t *X_tmp = 0;
	cudaMalloc(&X_tmp, sizeof(data_t) * in_len);
	cudaMemcpy(X_tmp, X, sizeof(data_t) * in_len, cudaMemcpyDefault);
	
	data_t *Y = policy->inference_execute(X_tmp, cuda_pointer_output);

	// Insert to memory
	size_t n_executions = mem.n_env_executions[env_i];
	mem.trajectory_inputs[env_i] =
		cuda_append_array(mem.trajectory_inputs[env_i], in_len * n_executions,
						  X_tmp, in_len, true);
	mem.trajectory_outputs[env_i] =
		cuda_append_array(mem.trajectory_outputs[env_i], out_len * n_executions,
						  Y, out_len, true);
	
	mem.current_value_internal_states[env_i] = value_function->get_hidden_state();
	mem.current_internal_states[env_i] = policy->get_hidden_state();

	data_t *out = alloc_output(out_len, output_kind);
	if (out) cudaMemcpy(out, Y, sizeof(data_t) * out_len, cudaMemcpyDefault);

	cudaFree(X_tmp);
	cudaFree(Y);
	*mem_pntr = mem;
	return out;
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
	data_t total_kl_divergence = 0;
	bool stop = false;
	for (size_t i = 0; i < hyperparameters.max_kl_divergence_threshold && !stop; i++)
	{
		data_t* execution_values = 0;
		data_t* activations = 0;
		data_t* Y = 0;
		policy_clone->training_execute(data_point_count, X, &Y, cuda_pointer_output, &execution_values, &activations);
		if (!Y) throw;

		size_t costs_len = neuron_count * data_point_count;
		data_t *costs = 0;
		cudaMalloc(&costs, sizeof(data_t) * costs_len);
		cudaMemset(costs, 0, sizeof(data_t) * costs_len);

		data_t kl_divergence = PPO_derivative(
			data_point_count, output_len, neuron_count,
			trajectory_Y, Y, advantages, costs, output_activations_start,
			hyperparameters.clip_ratio, hyperparameters.max_kl_divergence_threshold
		);
		total_kl_divergence += kl_divergence;
#ifdef DEBUG
		if (kl_divergence > .02) printf("KL divergence too high! %.3f\n", kl_divergence);
#endif
		stop = fabs(total_kl_divergence / (i + 1)) > hyperparameters.max_kl_divergence_threshold;
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
		collected_gradients = cuda_append_array(collected_gradients, gradient_count * data_point_count * i,
			gradients, gradient_count * data_point_count, true);
		cudaFree(gradients);
	}
	delete policy_clone;
	for (size_t i = 0; i < data_point_count; i++)
		policy->subtract_gradients(collected_gradients, gradient_count * i, hyperparameters.policy);
	cudaFree(collected_gradients);

}

void PPO_train(
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO_internal_memory *mem_pntr
)
{
	PPO_internal_memory mem = *mem_pntr;
	bool is_recurrent = value_function->is_recurrent() || policy->is_recurrent();

	std::vector<data_t *> advantages;
	for (size_t i = 0; i < mem.n_env; i++)
	{
		advantages.push_back(calculate_advantage(
			hyperparameters.steps_before_training,
			value_function, mem.trajectory_inputs[i], hyperparameters.GAE, false, false,
			mem.rewards[i], false, false
		));
	}

	size_t steps_per_env = hyperparameters.steps_before_training;
	size_t env_n = hyperparameters.vecenvironment_count;
	size_t total_execution_count = env_n * steps_per_env;
	if (is_recurrent)
	{
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
				mem.trajectory_inputs[i], policy->get_input_length() * steps_per_env,
				true
			);

		}
		// TODO: add shuffling

		// Create training data

		size_t items_per_minib = steps_per_env / hyperparameters.mini_batch_count;
		size_t last_item_count = steps_per_env % hyperparameters.mini_batch_count;
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
}

void PPO_data_cleanup(PPO_internal_memory *mem_pntr)
{
	PPO_internal_memory mem = *mem_pntr;

	for (size_t i = 0; i < mem.n_env; i++)
	{
		cudaFree(mem.initial_value_internal_states[i]);
		mem.initial_value_internal_states[i] = 0;

		cudaFree(mem.initial_internal_states[i]);
		mem.initial_internal_states[i] = 0;

		mem.initial_value_internal_states[i]
			= cuda_clone_arr(mem.initial_value_internal_states[i], mem.value_internal_state_length);
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

data_t *PPO_execute_train(
	data_t *X,  size_t env_i,
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO_internal_memory *mem_pntr, output_pointer_type output_kind,
	bool delete_memory_before
)
{
	PPO_internal_memory mem = *mem_pntr;
	PPO_initialization(
		X, value_function, policy,
		hyperparameters, &mem,
		env_i
	);
	*mem_pntr = mem;

	// Execution
	data_t *out = PPO_execution(
		X, env_i, value_function, policy, &mem, output_kind, delete_memory_before
	);
	*mem_pntr = mem;

	// Training Checks
	bool start_training = 1;
	for (size_t i = 0; i < mem.n_env; i++)
	{
		if (mem.n_env_executions[i] >= hyperparameters.steps_before_training)
		{
			free_PPO_data(mem_pntr);
			throw "Irregular env_steps not Implemented";	
		}
		start_training = start_training && mem.n_env_executions[i] == hyperparameters.steps_before_training - 1;
	}
	*mem_pntr = mem;
	if (!start_training) return out;

	// Training loop

	// Mem cleanup (sets initial states to current states, does not free current states)	
	*mem_pntr = mem;
	PPO_data_cleanup(mem_pntr);
	return out;
}
