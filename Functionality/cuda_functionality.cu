#include "cuda_functionality.cuh"


__device__ data_t device_abs(data_t a)
{
	return a * (-1 + 2 * (a >= 0));
}

__device__ data_t device_min(data_t a, data_t b)
{
	return a * (a <= b) + b * (b < a);
}

__device__ data_t device_max(data_t a, data_t b)
{
	return a * (a >= b) + b * (b > a);
}

__device__ data_t device_closest_to_zero(data_t a, data_t b)
{
	short is_a_closer = abs(a) < abs(b);
	return a * is_a_closer + b * !is_a_closer;
}

__device__ data_t device_clip(data_t to_clip, data_t a, data_t b)
{
	data_t lower_clip = device_min(a, b);
	data_t upper_clip = device_max(a, b);

	return device_max(device_min(to_clip, upper_clip), lower_clip);
}

__device__ data_t device_random_uniform(curandStateXORWOW_t *state)
{
	return (curand(state) % (int)1e4) / 1e4;
}

__host__ data_t* alloc_output(size_t output_value_count, output_pointer_type output_type)
{
	data_t* output = 0;
	switch (output_type)
	{
	case cuda_pointer_output:
		cudaMalloc(&output, sizeof(data_t) * output_value_count);
		cudaMemset(output, 0, sizeof(data_t) * output_value_count);
		break;
	case host_cpp_pointer_output:
		output = new data_t[output_value_count];
		memset(output, 0, sizeof(data_t) * output_value_count);
		break;
	case no_output:
	default:
		return (0);
	}
	return output;
}

__device__ size_t get_tid()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void block_extract(
	size_t n_blocks, size_t block_value_count, size_t groups_per_sub_block, size_t extracted_groups_value_count,
	size_t block_count_gap_between_usable_blocks, size_t extracted_sub_block_value_start, size_t group_read_index,
	data_t *in_arr, data_t *out_arr
)
{
	size_t tid = get_tid();

	size_t out_len = n_blocks * groups_per_sub_block;
	if (tid >= out_len) return;

	size_t block_i = tid / groups_per_sub_block;
	size_t group_i = tid % groups_per_sub_block;

	size_t block_start = block_value_count * block_i + block_value_count * block_count_gap_between_usable_blocks * block_i;
	size_t read_i = block_start + extracted_sub_block_value_start + extracted_groups_value_count * group_i + group_read_index;

	out_arr[tid] = in_arr[read_i];
}

__global__ void block_insert(
	size_t n_blocks, size_t block_value_count, size_t groups_per_sub_block, size_t extracted_groups_value_count,
	size_t block_count_gap_between_usable_blocks, size_t extracted_sub_block_value_start, size_t group_read_index,
	data_t *in_arr, data_t *out_arr
)
{
	size_t tid = get_tid();

	size_t out_len = n_blocks * groups_per_sub_block;
	if (tid >= out_len) return;

	size_t block_i = tid / groups_per_sub_block;
	size_t group_i = tid % groups_per_sub_block;

	size_t block_start = block_value_count * block_i + block_value_count * block_count_gap_between_usable_blocks * block_i;
	size_t write_i = block_start + extracted_sub_block_value_start + extracted_groups_value_count * group_i + group_read_index;

	out_arr[write_i] = in_arr[tid];
}

__global__ void extract_execution_values(data_t *execution_values, data_t *write_arr, size_t neuron_count, size_t execution_values_per_neuron, size_t neuron_read_i)
{
	size_t tid = get_tid();
	if (tid >= neuron_count || !execution_values || !write_arr) return;

	write_arr[tid] = execution_values[execution_values_per_neuron * tid + neuron_read_i];
}

__host__ data_t *host_extract_execution_values(data_t *execution_values, size_t neuron_count, size_t execution_values_per_neuron, size_t neuron_read_i)
{
	data_t *out = 0;
	cudaMalloc(&out, sizeof(data_t) * neuron_count);
	extract_execution_values n_threads(neuron_count) (
		execution_values, out, neuron_count,
		execution_values_per_neuron, neuron_read_i
	);
	cudaDeviceSynchronize();
	return (out);
}

__global__ void set_execution_values(
	data_t *execution_values_layer_start, data_t *read_arr,
	size_t execution_values_per_neuron, size_t neuron_write_i,
	size_t neuron_count
)
{
	size_t tid = get_tid();
	if (tid >= neuron_count || !execution_values_layer_start || !read_arr) return;

	size_t write_i = execution_values_per_neuron * tid + neuron_write_i;
	execution_values_layer_start[write_i] = read_arr[tid];
}

__global__ void exp_arr(data_t *arr, size_t arr_value_count)
{
	size_t tid = get_tid();
	if (tid >= arr_value_count || !arr) return;

	arr[tid] = expf(arr[tid]);
}

__global__ void reset_NaNs(field_t *array, field_t reset_value, size_t length)
{
#ifndef DEBUG
	size_t tid = get_tid();
	if (tid >= length)
		return;

	field_t value = array[tid];
	if (value != value)
		array[tid] = reset_value;
#endif
}

__global__ void mutate_field_array(
	field_t* array, size_t length, 
	float mutation_chance, float max_mutation, 
	float* triple_length_normalized_random_arr
)
{
	size_t tid = get_tid();
	if (tid >= length) return;

	array[tid] += triple_length_normalized_random_arr[tid] * max_mutation * (triple_length_normalized_random_arr[tid + length] < mutation_chance);
	array[tid] *= 1 - 2 * (triple_length_normalized_random_arr[tid + length * 2] < .5);
}

__global__ void global_initialize_parameters(field_t *params, size_t param_count, initialization_parameters init)
{
	size_t tid = get_tid();
	if (tid >= param_count) return;

	params[tid] = 0;

	curandStateXORWOW_t curand;
	curand_init(init.time, param_count, tid, &curand);
	switch (init.initialization)
	{
	case initialization_type::constant:
		params[tid] = init.constant.value_constant;
		break;
	case initialization_type::Xavier:
	{
		data_t factor = sqrtf(6.0 / (init.layer_n_inputs + init.layer_n_outputs));
		params[tid] = (device_random_uniform(&curand) - .5) * 2 * factor;
		break;
	}
	case initialization_type::central_limit:
		params[tid] = 
			sqrtf(-2 * logf(device_random_uniform(&curand))) * cosf(6.2831853 * device_random_uniform(&curand))
			* init.central_limit.std
			+ init.central_limit.mean;
		break;
	default:
		printf("Invalid initialization\n");
	}
}

__host__ void initialize_parameters(field_t **param_pntr, size_t param_count, initialization_parameters init)
{
	if (!param_pntr) throw;

	cudaMalloc(param_pntr, sizeof(field_t) * param_count);
	global_initialize_parameters n_threads(param_count) (*param_pntr, param_count, init);
	cudaDeviceSynchronize();
}

__host__ std::tuple<size_t *, size_t> cud_get_shuffled_indices(size_t stop, size_t start)
{
	size_t min_param = min(stop, start);
	size_t max_param = max(stop, start);
	start = min_param;
	stop = max_param;
	if (start == stop) return {0,0};

	size_t out_len = stop - start;
	size_t *out = 0;
	cudaMalloc(&out, sizeof(size_t) * out_len);
	write_indices n_threads(out_len) (
		out, out_len, start
	);
	cudaDeviceSynchronize();
	cuda_shuffle_inplace(out, out_len);
	return {out, out_len};
}
