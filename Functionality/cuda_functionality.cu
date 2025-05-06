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

__device__ size_t get_tid()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void count_value(size_t value, size_t* array, size_t array_length, unsigned int* output)
{
	size_t tid = get_tid();
	if (tid >= array_length) return;
	if (array[tid] != value) return;

	atomicAdd(output, 1);
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
