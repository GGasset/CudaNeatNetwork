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

__global__ void PPO_cost(
	data_t* activations, size_t activations_start,
	size_t neuron_count, size_t last_layer_activations_start, size_t output_length,
	data_t* costs, size_t costs_start,
	data_t* rewards
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;

	data_t ratio = 1;
	if (t) ratio = 
		activations[activations_start + neuron_count * t + last_layer_activations_start + tid] / 
		activations[activations_start + neuron_count * (t - 1) + last_layer_activations_start + tid];

	data_t reward = rewards[t];

	data_t clip = device_clip(ratio, 1 + .2, 1 - .2);
	data_t loss = device_min(ratio * reward, clip * reward);
	
	size_t cost_write = costs_start + neuron_count * t + last_layer_activations_start + tid;
	costs[cost_write] = loss;
}
