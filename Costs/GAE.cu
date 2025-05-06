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
