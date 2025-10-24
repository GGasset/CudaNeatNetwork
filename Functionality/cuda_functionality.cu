#include "cuda_functionality.cuh"

/*__global__ template void apply_to_array<typename t>(t* array, size_t array_length, std::function<bool(t, t)> if_function, t right_if_function_parameter, std::function<t(t)> to_apply)
{
	size_t tid = get_tid();
	if (tid >= array_length) return;

	t value = arrray[tid];
	if (if_function(value, right_if_function))
		array[tid] = to_apply(value);
}*/



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

__global__ void reset_NaNs(field_t *array, field_t reset_value, size_t length)
{
	size_t tid = get_tid();
	if (tid >= length)
		return;

	field_t value = array[tid];
	if (value != value)
		array[tid] = reset_value;
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
