
#include "PPO.cuh"

data_t *PPO_execute(
	data_t *X, size_t X_len, NN *policy,
	PPO_memory &mem, PPO_hyperparameters parameters, bool is_X_on_host, arr_location output_location,
	bool delete_memory_before
)
{
    if (policy->is_recurrent()) throw;
    if (!mem.was_reward_added && mem.n_executions) return 0;
    if (
        !X || X_len != policy->get_input_length() * parameters.vecenvironment_count
        || parameters.mini_batch_count > parameters.steps_before_training * parameters.vecenvironment_count
    ) return 0;

    data_t *activations = 0;
    data_t *execution_values = 0;

    data_t *device_Y = policy->execute(
        parameters.vecenvironment_count, policy->is_recurrent() * mem.n_executions,
        X, X_len, is_X_on_host, device_arr,
        &activations, &execution_values, delete_memory_before,
        mem.last_execution_values, 1
    );
    if (policy->is_recurrent())
    {
        cudaFree(mem.last_execution_values);
        mem.last_execution_values = execution_values;

        cudaFree(mem.last_activations);
        mem.last_activations = activations;
    }
    else cudaFree(execution_values);
    cudaFree(activations);
    if (!device_Y) return 0;

    size_t in_count = policy->get_input_length() * parameters.vecenvironment_count;
    mem.inputs = cuda_append_array(mem.inputs, in_count * mem.n_executions, X, in_count, true, true);

    size_t out_count = policy->get_output_length() * parameters.vecenvironment_count;
    mem.outputs = cuda_append_array(mem.outputs, out_count * mem.n_executions, device_Y, out_count, true);

    mem.n_executions++;
    mem.was_reward_added = false;

    data_t *Y = alloc_output(out_count, output_location);
    if (Y) cudaMemcpy(Y, device_Y, sizeof(data_t) * out_count, cudaMemcpyDefault);

    return Y;
}

void non_recurrent_PPO_minibatch(
    data_t *inputs, data_t *outputs, data_t *advantages, size_t n_executions,
    NN *policy, PPO_hyperparameters parameters,
    size_t minibatch_start_i, size_t minibatch_len
)
{
    size_t inputs_start = minibatch_start_i * policy->get_input_length();
    size_t inputs_len = minibatch_len * policy->get_input_length();

    size_t outputs_start = minibatch_start_i * policy->get_output_length();
    size_t outputs_len = minibatch_len * policy->get_output_length();

    int stop = 0;
    for (size_t i = 0; i < parameters.max_training_steps && !stop; i++)
    {
        data_t *activations = 0;
        data_t *execution_values = 0;

        data_t *current_outputs = policy->execute(
            n_executions, 0, inputs + inputs_start, inputs_len, false, device_arr, 
            &activations, &execution_values
        );

        // Get output cost
        data_t kl_divergence_aproximation = 0;
        data_t *output_cost = PPO_derivative(
            minibatch_len, policy->get_output_length(),
            outputs, current_outputs, false, advantages,
            parameters.clip_ratio, kl_divergence_aproximation
        );

        if (kl_divergence_aproximation > parameters.max_kl_divergence_threshold) stop = true;
        
        data_t *gradients = policy->backpropagate(
            n_executions, 1, output_cost, outputs_len, activations, execution_values, parameters.policy
        );

        policy->subtract_gradients(n_executions, 1, gradients, parameters.policy);
        
        cudaFree(activations);
        cudaFree(execution_values);
        cudaFree(output_cost);
        cudaFree(gradients);
    }
}

// Parameters to this function must have the timesteps of the same execution line together
void non_recurrent_PPO_train(
    data_t *inputs, data_t *outputs, data_t *advantages, size_t n_executions,
    NN *policy, PPO_hyperparameters parameters
)
{
    size_t total_executions = n_executions * parameters.vecenvironment_count;

    size_t minibatch_execution_i_start = 0;
    for (size_t minibatch_i = 0; minibatch_i < parameters.mini_batch_count; minibatch_i++)
    {
        size_t executions_in_minibatch = total_executions / parameters.mini_batch_count;
        if (minibatch_execution_i_start + executions_in_minibatch >= total_executions)
            executions_in_minibatch = total_executions - minibatch_execution_i_start;

        non_recurrent_PPO_minibatch(
            inputs, outputs, advantages, executions_in_minibatch,
            policy, parameters, minibatch_execution_i_start, executions_in_minibatch
        );

        minibatch_execution_i_start += executions_in_minibatch;
    }
}

int add_rewards(
	data_t *rewards, size_t rewards_len,
	NN *value_function, NN *policy, PPO_memory &mem, PPO_hyperparameters parameters
)
{
    if (!rewards || rewards_len != parameters.vecenvironment_count || mem.was_reward_added) return true;

    mem.rewards = cuda_append_array(mem.rewards, rewards_len * mem.n_executions, rewards, rewards_len, true, true);
    mem.was_reward_added = true;

    if (mem.n_executions < parameters.steps_before_training) return 0;
    
    // Training
    std::cout << "\nTraining..." << std::endl;

    size_t total_execution_count = mem.n_executions * parameters.vecenvironment_count;

    size_t total_input_count = policy->get_input_length() * total_execution_count;
    data_t *inputs = cudaCalloc<data_t>(total_input_count);

    size_t total_output_count = policy->get_output_length() * total_execution_count;
    data_t *outputs = cudaCalloc<data_t>(total_output_count);

    continuize_arrs n_threads(total_input_count) (
        mem.inputs, inputs, total_input_count,
        mem.n_executions, parameters.vecenvironment_count,
        policy->get_input_length()
    );

    continuize_arrs n_threads(total_output_count) (
        mem.outputs, outputs, total_output_count,
        mem.n_executions, parameters.vecenvironment_count,
        policy->get_output_length()
    );
    cudaDeviceSynchronize();

    data_t *advantages = 
        get_advantages(
            parameters.vecenvironment_count, mem.n_executions, value_function,
            inputs, total_input_count, parameters.GAE, mem.rewards
        );

    auto [shuffled_keys, key_arr_len] = cud_get_shuffled_indices(total_execution_count);

    cuda_sort_by_key(&inputs, shuffled_keys, key_arr_len, policy->get_input_length());
    cuda_sort_by_key(&outputs, shuffled_keys, key_arr_len, policy->get_output_length());
    cuda_sort_by_key(&advantages, shuffled_keys, key_arr_len, 1);

    if (!policy->is_recurrent() && !value_function->is_recurrent())
        non_recurrent_PPO_train(
            inputs, outputs, advantages, mem.n_executions,
            policy, parameters
        );
    else
        throw;

    mem.deallocate(false);
    cudaFree(inputs);
    cudaFree(outputs);
    cudaFree(advantages);

    return 0;
}

void PPO_memory::deallocate(bool free_memory_execution_values)
{
    cudaFree(inputs);
    inputs = 0;
    cudaFree(outputs);
    outputs = 0;
    cudaFree(rewards);
    rewards = 0;

    if (free_memory_execution_values)
    {
        cudaFree(last_execution_values);
        last_execution_values = 0;
    }

    n_executions = 0;
}
