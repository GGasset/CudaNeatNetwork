
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
		|| mem.n_env_executions[env_i] != mem.add_reward_calls_n[env_i]
		|| !hyperparameters.steps_before_training
		|| !hyperparameters.max_training_steps
		|| !hyperparameters.mini_batch_size
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
