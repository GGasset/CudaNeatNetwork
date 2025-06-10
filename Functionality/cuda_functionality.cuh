#pragma once

#include <functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

#include "functionality.h"

//__global__ template void apply_to_array<typename t>(t* array, size_t array_length, std::function<bool(t, t)> if_function, t right_if_function_parameter, std::function<t(t)> to_apply);
__device__ data_t device_min(data_t a, data_t b);
__device__ data_t device_max(data_t a, data_t b);
__device__ data_t device_closest_to_zero(data_t a, data_t b);
__device__ data_t device_clip(data_t to_clip, data_t a, data_t b);

/// <summary>
/// Calculates linear thread_id up to blockIdx.x [inclusive]
/// </summary>
/// <returns></returns>
__device__ size_t get_tid();

template<typename T, typename t>
__global__ void multiply_array(T* arr, size_t arr_value_count, t multiply_by_value)
{
	size_t tid = get_tid();
	if (tid >= arr_value_count) return;

	arr[tid] *= multiply_by_value;
}

template<typename T, typename t>
__global__ void element_wise_multiply(T* to_multiply, t* multiplier_arr, size_t min_arr_value_count)
{
	size_t tid = get_tid();
	if (tid >= min_arr_value_count) return;

	to_multiply[tid] *= multiplier_arr[tid];
}

template<typename T, typename t>
__global__ void add_to_array(T* arr, size_t arr_value_count, t to_add)
{
	size_t tid = get_tid();
	if (tid >= arr_value_count) return;

	arr[tid] += to_add;
}

template <typename T, typename t>
__global__ void logical_copy(T* dst, size_t dst_len, t* src, size_t src_len)
{
	size_t tid = get_tid();
	if (tid >= device_min(dst_len, src_len)) return;

	dst[tid] = src[tid];
}

template<typename T>
__global__ void count_value(T value, T* array, size_t array_length, unsigned int* output)
{
	size_t tid = get_tid();
	if (tid >= array_length) return;
	if (array[tid] != value) return;

	atomicAdd(output, 1);
}

__global__ void reset_NaNs(field_t *array, field_t reset_value, size_t length);

__global__ void mutate_field_array(
	field_t* array, size_t length,
	float mutation_chance, float max_mutation,
	float* triple_length_normalized_random_arr
);

template<typename T>
__host__ T* cuda_realloc(T* old, size_t old_len, size_t new_len, bool free_old)
{
	T* out = 0;
	cudaMalloc(&out, sizeof(T) * new_len);
	cudaMemset(out, 0, sizeof(T) * new_len);
	cudaMemcpy(out, old, sizeof(T) * h_min(old_len, new_len), cudaMemcpyDeviceToDevice);
	if (free_old)
		cudaFree(old);
	return out;
}

template<typename T>
__host__ T* cuda_remove_elements(T* old, size_t len, size_t remove_start, size_t remove_count, bool free_old)
{
	if (!len || !old) return (0);

	remove_start = h_min(len, remove_start);
	remove_count = h_min(len - remove_start, remove_count);

	T* out = 0;
	cudaMalloc(&out, sizeof(T) * (len - remove_count));
	if (!out) return (0);
	cudaMemset(out, 0, sizeof(T) * (len - remove_count));

	if (remove_start)
		cudaMemcpy(out, old, sizeof(T) * (remove_start), cudaMemcpyDeviceToDevice);
	if (!(remove_start + remove_count >= len))
		cudaMemcpy(out + remove_start, old + remove_start + remove_count, sizeof(T) * (len - remove_count - remove_start), cudaMemcpyDeviceToDevice);
	return out;
}

template<typename T, typename t>
__host__ T* cuda_remove_occurrences(t* compare_arr, t to_delete_number, T* to_update_arr, size_t arrs_len, bool free_updated)
{
	T* host_arr = new T[len];
	cudaMemcpy(host_arr, old, sizeof(T) * len, cudaMemcpyDeviceToHost);

	t* host_compare_arr = new t[len];
	cudaMemcpy(host_compare_arr, compare_arr, sizeof(T) * len, cudaMemcpyDeviceToHost);

	std::vector<T> parsed_vector = std::vector<T>();
	for (size_t i = 0; i < len; i++)
		if (host_compare_arr[i] != to_delete_number) 
			parsed_vector.push_back(host_arr[i]);

	delete[] host_compare_arr;
	host_compare_arr = 0;
	delete[] host_arr;
	host_arr = 0;

	T* out = 0;
	cudaMalloc(&out, sizeof(T) * parsed_vector.size());
	cudaMemcpy(out, parsed_vector.data(), sizeof(T) * parsed_vector.size(), cudaMemcpyHostToDevice);

	if (free_old) cudaFree(old);
	return out;
}

template<typename T>
__host__ void print_array(T* arr, size_t arr_len)
{
	T* host_arr = new T[arr_len];
	cudaMemcpy(host_arr, arr, sizeof(T) * arr_len, cudaMemcpyDefault);

	for (size_t i = 0; i < arr_len; i++) printf("%.2f ", (float)(host_arr[i]));
	printf("\n");

	delete[] host_arr;
}

template<typename T>
__host__ T* cuda_push_back(T *old, size_t old_len, T new_last, bool free_old)
{
	T* out = cuda_realloc(old, old_len, old_len + 1, free_old);

	T tmp = new_last;
	cudaMemcpy(out + old_len, &tmp, sizeof(T), cudaMemcpyHostToDevice);
	return out;
}
