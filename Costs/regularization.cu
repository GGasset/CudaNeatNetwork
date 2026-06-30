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
	
	data_t l2 = PRAM_reduce_add(gradients_copy, gradient_count);
	cudaFree(gradients_copy);
	if (l2 <= hyperparameters.global_gradient_clip) return;

	data_t to_multiply_by = hyperparameters.global_gradient_clip / (l2 + 1e-7);
	
	multiply_array n_threads(gradient_count) (gradients, gradient_count, to_multiply_by);
	cudaDeviceSynchronize();
}

void value_normalizer::update_mean_std()
{
	mean = sum / n;
	std = sqrt(sum_of_squares / n - mean * mean);
}

data_t value_normalizer::normalize_val(data_t v)
{
    return (v - mean) / std;
}

data_t *value_normalizer::incoming_vals(
	data_t *vals, size_t n_vals, bool are_vals_on_host, arr_location _return_location
)
{
	if (are_vals_on_host)
		vals = cuda_clone_arr(vals, n_vals);

    n += n_vals;
	sum += PRAM_reduce_add(vals, n_vals);

	data_t *squared_vals = cuda_clone_arr(vals, n_vals);
	element_wise_multiply n_threads(n_vals) (squared_vals, squared_vals, n_vals);
	cudaDeviceSynchronize();
	sum_of_squares += PRAM_reduce_add(squared_vals, n_vals);
	cudaFree(squared_vals);


	update_mean_std();

	data_t *out = cuda_clone_arr(vals, n_vals);
	normalize_arr n_threads(n_vals) (out, n_vals, mean, std);
	
	if (are_vals_on_host)
		cudaFree(vals);
		
	cudaDeviceSynchronize();
	return out;
}

data_t value_normalizer::incoming_val(data_t v)
{
	sum += v;
	sum_of_squares += v * v;
	n++;

	update_mean_std();

    return normalize_val(v);
}
