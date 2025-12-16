#include "regularization.cuh"

__global__ void global_entropy_regularization(
	size_t t_count, size_t neuron_count, size_t output_len,
	data_t* activations, size_t last_layer_activations_start,
	entropy_bonus_hyperparameters hyperparameters,
	data_t *costs
)
{
	size_t tid = get_tid();
	if (!costs || tid >= t_count * output_len) return;

	size_t t = tid / output_len;
	size_t output_i = tid % output_len;

	size_t neuron_i = t * neuron_count + last_layer_activations_start + output_i;

	data_t activation = activations[neuron_i];
	data_t abs_activation = abs(activation) + 1E-10;

	data_t entropy_bonus = (1 - 2 * (activation < 0)) * hyperparameters.entropy_coefficient * ((logf(abs_activation) + 1) / logf(10));
	atomicAdd(costs + neuron_i, entropy_bonus);
}

void entropy_regularization(
	size_t t_count, size_t neuron_count, size_t output_len,
	data_t* costs, data_t* activations, size_t last_layer_activations_start,
	entropy_bonus_hyperparameters hyperparameters
)
{
	if (!hyperparameters.active || !hyperparameters.entropy_coefficient) return;

	size_t block_count = t_count * output_len;
	global_entropy_regularization kernel(block_count / 32 + (block_count % 32 > 0), 32) (
		t_count, neuron_count, output_len,
		activations, last_layer_activations_start,
		hyperparameters,
		costs
	);
	cudaDeviceSynchronize();
}

__host__ void global_gradient_clip(data_t *gradients, size_t gradient_count, gradient_hyperparameters hyperparameters)
{
	if (!hyperparameters.global_gradient_clip) return;

	data_t *gradients_copy = cuda_clone_arr(gradients, gradient_count);
	element_wise_multiply n_threads(gradient_count) (gradients_copy, gradients_copy, gradient_count);
	cudaDeviceSynchronize();
	
	data_t l2 = cuda_sum<data_t, data_t>(gradients_copy, gradient_count);
	cudaFree(gradients_copy);
	if (l2 <= hyperparameters.global_gradient_clip) return;

	data_t to_multiply_by = sqrt(l2 * (1 / abs(hyperparameters.global_gradient_clip) + 1e-5));
	to_multiply_by = 1 / (to_multiply_by + 1e-5);
	
	multiply_array n_threads(gradient_count) (gradients, gradient_count, to_multiply_by);
	cudaDeviceSynchronize();
}
