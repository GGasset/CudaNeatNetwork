#include "costs.cuh"

__global__ void MSE_derivative(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* costs, size_t costs_start,
	data_t* Y_hat, size_t output_length
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;

	data_t predicted = activations[activations_start + neuron_count * t + last_layer_activations_start + tid];
	data_t Y = Y_hat[output_length * t + tid]; 
	//data_t derivative = -2 * (Y_hat[output_length * t + tid] - activations[activations_start + neuron_count * t + last_layer_activations_start + tid]);
	data_t derivative = 2 * (predicted - Y);
	costs[costs_start + t * neuron_count + last_layer_activations_start + tid] = derivative;
}

__global__ void MSE_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* Y_hat, size_t output_length,
	data_t* cost_write
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;

	data_t predicted = activations[activations_start + neuron_count * t + last_layer_activations_start + tid];
	data_t Y = Y_hat[output_length * t + tid];
	data_t error = Y - predicted;
	error *= error;
	atomicAdd(cost_write, error);
}

__global__ void log_likelyhood_derivative(
	data_t* activations, size_t activations_start,
	size_t neuron_count, size_t last_layer_activations_start, size_t output_length,
	data_t* costs, size_t costs_start,
	data_t* rewards
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;

	data_t reward = rewards[t];
	data_t activation = neuron_count * t + last_layer_activations_start + tid;
	data_t cost_derivative = -(reward / activation);


	size_t cost_write = costs_start + neuron_count * t + last_layer_activations_start + tid;
	costs[cost_write] = cost_derivative;
}

__global__ void log_likelyhood_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* rewards, size_t output_length,
	data_t* cost
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;
	
	data_t reward = rewards[t];
	data_t prediction = activations[activations_start + neuron_count * t + last_layer_activations_start + tid];
	data_t output = -log(prediction) * reward;
	
	atomicAdd(cost, output);
}

__global__ void device_PPO_derivative(
	size_t t_count, size_t output_len, size_t neuron_count,
	data_t *outputs, data_t *current_outputs, data_t *advantages,
	data_t *costs, size_t last_layer_activations_start,
	data_t clip_ratio, data_t *summed_kl_divergence
)
{
	size_t tid = get_tid();
	if (tid >= output_len) return;
	size_t t = blockIdx.y;

	data_t advantage = advantages[t];
	data_t initial_output = outputs[t * output_len + tid];
	data_t output = current_outputs[t * output_len + tid];
	data_t ratio = output / initial_output;

	size_t cost_write_i = t * neuron_count + last_layer_activations_start + tid;
	data_t cost_derivative = -((advantage * output) / (initial_output * initial_output));
	cost_derivative *= 1 + clip_ratio > ratio && 1 - clip_ratio < ratio;

	costs[cost_write_i] = cost_derivative;

	atomicAdd(summed_kl_divergence, initial_output * log(initial_output / output));
}

__host__ int PPO_derivative(
	size_t t_count, size_t output_len, size_t neuron_count,
	data_t* trajectory_outputs, data_t* current_outputs, data_t* advantages,
	data_t* costs, size_t last_layer_activations_start,
	data_t clip_ratio, data_t kl_divergence_threshold
)
{
	data_t* total_kl_divergence = 0;
	cudaMalloc(&total_kl_divergence, sizeof(data_t));
	cudaMemset(total_kl_divergence, 0, sizeof(data_t));
	if (!total_kl_divergence) return 1;

	dim3 gridDim(output_len / 32 + (output_len % 32 > 0), t_count);
	device_PPO_derivative kernel(gridDim, 32) (
		t_count, output_len, neuron_count,
		trajectory_outputs, current_outputs, advantages,
		costs, last_layer_activations_start,
		clip_ratio, total_kl_divergence
	);
	cudaDeviceSynchronize();

	data_t host_total_kl_divergence = 0;
	cudaMemcpy(&host_total_kl_divergence, total_kl_divergence, sizeof(data_t), cudaMemcpyDeviceToHost);
	cudaFree(total_kl_divergence);

	data_t mean_kl_divergence = host_total_kl_divergence / t_count / output_len;
	return mean_kl_divergence >= kl_divergence_threshold;
}
