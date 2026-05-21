
#include "PPO.cuh"

data_t *PPO_execute(
	data_t *X, size_t X_len, NN *policy,
	PPO_memory &mem, PPO_hyperparameters parameters, bool is_X_on_host, arr_location output_location,
	bool delete_memory_before
)
{
    if (policy->is_recurrent()) throw;
    if (!mem.was_reward_added) return 0;
    if (!X || X_len != policy->get_input_length() * parameters.vecenvironment_count) return 0;

    data_t *activations = 0;
    data_t *execution_values = 0;

    data_t *device_Y = policy->execute(
        parameters.vecenvironment_count, mem.n_executions,
        X, X_len, is_X_on_host, device_arr,
        &activations, &execution_values, delete_memory_before,
        mem.execution_values, 1
    );
    if (policy->is_recurrent())
    {
        cudaFree(mem.execution_values);
        mem.execution_values = execution_values;
    }
    else cudaFree(execution_values);
    cudaFree(activations);
    if (!device_Y) return 0;

    size_t in_count = policy->get_input_length() * parameters.vecenvironment_count;
    mem.inputs = cuda_append_array(mem.inputs, in_count * mem.n_executions, X, in_count, true);

    size_t out_count = policy->get_output_length() * parameters.vecenvironment_count;
    mem.outputs = cuda_append_array(mem.outputs, out_count * mem.n_executions, device_Y, out_count, true);

    mem.n_executions++;
    mem.was_reward_added = false;

    data_t *Y = alloc_output(out_count, output_location);
    if (Y) cudaMemcpy(Y, device_Y, sizeof(data_t) * out_count, cudaMemcpyDefault);

    return Y;
}

// Parameters to this function must have the timesteps of the same execution line together
void non_recurrent_PPO_train(
    data_t *inputs, data_t *outputs, data_t *advantages, size_t n_executions,
    NN *value_function, NN *policy, PPO_hyperparameters parameters
)
{
    size_t total_execution = n_executions * parameters.vecenvironment_count;

    auto [shuffled_keys, key_arr_len] = cud_get_shuffled_indices(total_execution);
}

int add_rewards(
	data_t *rewards, size_t rewards_len,
	NN *value_function, NN *policy, PPO_memory &mem, PPO_hyperparameters parameters
)
{
    if (!rewards || rewards_len != parameters.vecenvironment_count || mem.was_reward_added) return true;

    mem.rewards = cuda_append_array(mem.rewards, rewards_len * mem.n_executions, rewards, rewards_len, true);
    mem.was_reward_added = true;

    if (mem.n_executions < parameters.steps_before_training) return 0;
    
    // Training

    size_t total_execution_count = mem.n_executions * parameters.vecenvironment_count;

    size_t total_input_count = policy->get_input_length() * total_execution_count;
    data_t *continuized_inputs = cudaCalloc<data_t>(total_input_count);

    size_t total_output_count = policy->get_output_length() * total_execution_count;
    data_t *continuized_outputs = cudaCalloc<data_t>(total_output_count);

    continuize_arrs n_threads(total_input_count) (
        mem.inputs, continuized_inputs, total_input_count,
        mem.n_executions, parameters.vecenvironment_count,
        policy->get_input_length()
    );

    continuize_arrs n_threads(total_output_count) (
        mem.outputs, continuized_outputs, total_output_count,
        mem.n_executions, parameters.vecenvironment_count,
        policy->get_output_length()
    );
    cudaDeviceSynchronize();

    data_t *advantages = 
        get_advantages(
            parameters.vecenvironment_count, mem.n_executions, value_function,
            continuized_inputs, total_input_count, parameters.GAE, mem.rewards
        );

    if (!policy->is_recurrent() && !value_function->is_recurrent())
        non_recurrent_PPO_train(
            continuized_inputs, continuized_outputs, advantages, mem.n_executions,
            value_function, policy, parameters
        );
    else
        throw;

    mem.deallocate(false);
    cudaFree(continuized_inputs);
    cudaFree(continuized_outputs);
    cudaFree(advantages);
}
