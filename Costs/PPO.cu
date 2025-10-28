
# include "PPO.cuh"

data_t *PPO_execute_train(
	data_t *X, size_t env_i,
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO_internal_memory *mem_pntr, output_pointer_type output_kind
)
{
	// Checks
	PPO_internal_memory mem = *mem_pntr;
	if (!hyperparameters.vecenvironment_count
	 || !hyperparameters.steps_before_training
	 || !hyperparameters.max_training_steps
	 || !hyperparameters.mini_batch_size)
		return (data_t *)free_PPO_data(mem_pntr);
	
	// Initialization
	if (!mem.n_env)
	{
		size_t n_env = hyperparameters.vecenvironment_count;
		mem_pntr->n_env = n_env;

		mem_pntr->initial_internal_states.resize(n_env, 0);
		mem_pntr->current_internal_states.resize(n_env, 0);

		mem_pntr->trajectory_inputs.resize(n_env, 0);
		mem_pntr->trajectory_outputs.resize(n_env, 0);

		mem_pntr->rewards.resize(n_env, 0);
		mem_pntr->add_reward_calls_n.resize(n_env, 0);

		mem_pntr->n_env_executions.resize(n_env, 0);

		size_t value_func_state_value_count = 0;
		data_t *value_func_state = value_function->get_hidden_state(&value_func_state_value_count);

		size_t policy_state_value_count = 0;
		data_t *policy_state = policy->get_hidden_state(&policy_state_value_count);
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
		mem = *mem_pntr;
	}
	if (mem.is_initialized) // states should be initialized and n_env, everything else should not
	{

	}

	if (mem.n_env != hyperparameters.vecenvironment_count) throw;

	// Execution
	data_t *out = 0;

	// Check if add_reward was called after last PPO_execute_train call

	// Insert to memory

	// Training Checks
	bool start_training = 1;
	for (size_t i = 0; i < mem.n_env; i++)
	{
		if (mem.n_env_executions[i] >= hyperparameters.steps_before_training)
		{
			free_PPO_data(mem_pntr);
			throw "Irregular_env_steps not added";	
		}
		start_training = start_training && mem.n_env_executions[i] == hyperparameters.steps_before_training - 1;
	}
	if (!start_training) return out;

	// Training loop

	// Mem cleanup (sets initial states to current states, does not free current states)
	return out;
}
