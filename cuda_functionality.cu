#include "cuda_functionality.cuh"

__global__ void multiply_array(float* arr, size_t arr_value_count, float multiply_by_value)
{
	size_t tid = get_tid();
	if (tid >= arr_value_count) return;

	arr[tid] *= multiply_by_value;
}

__device__ data_t device_abs(data_t a)
{
	return a * (-1 + 2 * (a >= 0));
}

__device__ data_t device_min(data_t a, data_t b)
{
	return a * (a <= b) + b * (b < a);
}

__device__ data_t device_closest_to_zero(data_t a, data_t b)
{
	short is_a_closer = abs(a) < abs(b);
	return a * is_a_closer + b * !is_a_closer;
}

__device__ size_t get_tid()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
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
