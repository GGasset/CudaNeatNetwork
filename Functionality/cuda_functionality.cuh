
#pragma once

#include "curand.h"
#include <functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"
#include "kernel_macros.h"

#include "functionality.h"
#include "NN_enums.h"

__device__ data_t device_min(data_t a, data_t b);
__device__ data_t device_max(data_t a, data_t b);
__device__ data_t device_closest_to_zero(data_t a, data_t b);
__device__ data_t device_clip(data_t to_clip, data_t a, data_t b);

__host__ data_t* alloc_output(size_t output_value_count, output_pointer_type output_type);

/// <summary>
/// Calculates linear thread_id up to blockIdx.x [inclusive]
/// </summary>
/// <returns></returns>
__device__ size_t get_tid();

__global__ void extract_execution_values(
	data_t *execution_values_layer_start, data_t *write_arr, size_t layer_length,
	size_t execution_values_per_neuron, size_t neuron_read_i
);

__host__ data_t *host_extract_execution_values(
	data_t *execution_values_layer_start,  size_t neuron_count,
	size_t execution_values_per_neuron, size_t neuron_read_i
);

__global__ void set_execution_values(
	data_t *execution_values_layer_start, data_t *read_arr,
	size_t execution_values_per_neuron, size_t neuron_write_i,
	size_t neuron_count
);

template<typename T, typename param_t, typename return_t>
__global__ void apply_func(T *arr, size_t arr_len, return_t (*func)(param_t x))
{
	size_t tid = get_tid();
	if (tid >= arr_len) return;

	arr[tid] = (T)func((param_t)arr[tid]);
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

// Only the values whose index is less than the min len
// Parameter Write_arr may be a or b
// Parameters Invert [a,b]: the corresponding array values will be multiplied by -1 in the sum
template<typename T, typename t>
__global__ void add_arrays(T *write_arr, T *a, t *b, size_t a_len, size_t b_len, bool invert_a = false, bool invert_b = false)
{
	size_t tid = get_tid();
	if (tid >= device_min(a_len, b_len)) return;

	write_arr[tid] += a[tid] * (1 - (2 * invert_a)) + b[tid] * (1 - (2 * invert_b));
}

// write_arr requires an avalible length of n_inputs / 2 + n_inputs % 2
template<typename T>
__global__ void global_PRAM_reduce_add(T* input, T* write_arr, size_t n_inputs)
{
	size_t write_arr_len = n_inputs / 2 + n_inputs % 2;

	size_t tid = get_tid();
	if (tid >= write_arr_len || !input || !write_arr) return;
	if (tid == n_inputs / 2)
	{
		write_arr[tid] = input[n_inputs - 1];
		return;
	}
	write_arr[tid] = input[tid * 2] + input[tid * 2 + 1];
}

template<typename T>
__host__ T PRAM_reduce_add(T* arr, size_t arr_len, T *output_write = 0, bool is_arr_in_host = false)
{
	T *tmp = 0;

	cudaMalloc(&tmp, sizeof(T) * arr_len);

	cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
	if (is_arr_in_host) kind = cudaMemcpyHostToDevice;
	cudaMemcpy(tmp, arr, sizeof(T) * arr_len, kind);

	while (arr_len > 1)
	{
		if (!tmp) throw;
		size_t new_arr_len = arr_len / 2 + arr_len % 2;
		T* new_arr = 0;
		cudaMalloc(&new_arr, sizeof(T) * new_arr_len);
		global_PRAM_reduce_add n_threads(new_arr_len) (
			tmp, new_arr, arr_len
		);
		cudaDeviceSynchronize();
		cudaFree(tmp);
		tmp = new_arr;
		arr_len = new_arr_len;
	}
	if (output_write) cudaMemcpy(output_write, tmp, sizeof(T), cudaMemcpyDefault);

	T out;
	cudaMemcpy(&out, tmp, sizeof(T), cudaMemcpyDeviceToHost);
	return out;
}

// 00000 00000 00000
//  012   345   678
//   01   1 2   34
template<typename T>
__global__ void global_multi_PRAM_add(T *input, size_t arr_count, size_t arr_len, T *write_arr)
{
	size_t tid = get_tid();
	size_t new_arr_len = arr_len / 2 + (arr_len % 2);
	if (tid >= arr_count * new_arr_len) return;

	size_t arr_i = tid / new_arr_len;

	size_t write_elem_i = tid % new_arr_len;
	if (write_elem_i == new_arr_len - 1 && arr_len % 2)
	{
		size_t read_i = arr_len * arr_i + arr_len - 1;
		write_arr[tid] = input[read_i];
		return;
	}

	size_t read_i = tid * 2 + arr_i * (arr_len % 2);
	write_arr[tid] = input[read_i] + input[read_i + 1];
}

template<typename T>
__host__ T *multi_PRAM_add(T *arrs, size_t arr_len, size_t arr_count = 1)
{
	if (!arr_count || !arr_len || !arrs) return 0;
	T *tmp = 0;
	cudaMalloc(&tmp, sizeof(T) * arr_len * arr_count);
	cudaMemcpy(tmp, arrs, sizeof(T) * arr_len * arr_count, cudaMemcpyDefault);
	while (arr_len > 1)
	{
		size_t new_arr_len = arr_len / 2 + arr_len % 2;
		T *new_arr = 0;
		cudaMalloc(&new_arr, sizeof(T) * new_arr_len * arr_count);
		global_multi_PRAM_add n_threads(new_arr_len) (
			tmp, arr_count, arr_len, new_arr
		);
		cudaDeviceSynchronize();
		arr_len = new_arr_len;
		cudaFree(tmp);
		tmp = new_arr;
	}
	return tmp;
}

template<typename T>
__host__ T *cuda_clone_arr(T *arr, size_t arr_len)
{
	if (!arr) return 0;
	T *out = 0;
	cudaMalloc(&out, sizeof(T) * arr_len);
	cudaMemcpy(out, arr, sizeof(T) * arr_len, cudaMemcpyDefault);
	return out;
}

// Tries to copy, when src is copied and dst has free space, starts from the beginning
template<typename T>
__global__ void repetitive_copy(T *dst, size_t dst_len, T *src, size_t src_len)
{
	size_t tid = get_tid();
	if (tid >= dst_len || !dst || !src) return;

	dst[tid] = src[tid % src_len];
}

template <typename T, typename t>
__global__ void logical_copy(T* dst, size_t dst_len, t* src, size_t src_len)
{
	size_t tid = get_tid();
	if (tid >= device_min(dst_len, src_len)) return;

	dst[tid] = src[tid];
}

template<typename T>
__global__ void nullify_unless_equals(T* arr, size_t value_count, T no_nullify_value)
{
	size_t tid = get_tid();
	if (tid >= value_count) return;

	arr[tid] *= arr[tid] == no_nullify_value;
}

template<typename T>
__global__ void booleanize(T *arr, size_t value_count)
{
	size_t tid = get_tid();
	if (tid >= value_count) return;

	arr[tid] = arr[tid] != 0;
}

template<typename T>
__host__ size_t count_value(T value, T* array, size_t array_length)
{
	T *tmp = cuda_clone_arr(array, array_length);
	if (!tmp) throw;

	nullify_unless_equals n_threads(array_length) (tmp, array_length, value);
	cudaDeviceSynchronize();
	booleanize n_threads(array_length) (tmp, array_length);
	cudaDeviceSynchronize();
	size_t out = (size_t)PRAM_reduce_add(tmp, array_length);
	cudaFree(tmp);
	return out;
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
	size_t new_size = sizeof(T) * new_len;
	cudaMalloc(&out, new_size);
	cudaMemset(out, 0, new_size);
	if (old)
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
	T* host_arr = new T[arrs_len];
	cudaMemcpy(host_arr, to_update_arr, sizeof(T) * arrs_len, cudaMemcpyDeviceToHost);

	t* host_compare_arr = new t[arrs_len];
	cudaMemcpy(host_compare_arr, compare_arr, sizeof(T) * arrs_len, cudaMemcpyDeviceToHost);

	std::vector<T> parsed_vector = std::vector<T>();
	for (size_t i = 0; i < arrs_len; i++)
		if (host_compare_arr[i] != to_delete_number) 
			parsed_vector.push_back(host_arr[i]);

	delete[] host_compare_arr;
	host_compare_arr = 0;
	delete[] host_arr;
	host_arr = 0;

	T* out = 0;
	cudaMalloc(&out, sizeof(T) * parsed_vector.size());
	cudaMemcpy(out, parsed_vector.data(), sizeof(T) * parsed_vector.size(), cudaMemcpyHostToDevice);

	if (free_updated) cudaFree(to_update_arr);
	return out;
}

inline int size_t_bigger_than_compare_func(size_t val, size_t second_arg)
{
	return val > second_arg;
}

template<typename T, typename t>
__host__ T* cuda_add_to_occurrences(t* compare_arr, int (*compare_func)(t val, t second_arg), t second_arg, T* to_update_arr, T to_add, size_t arrs_len, bool free_updated)
{
	T* host_arr = new T[arrs_len];
	cudaMemcpy(host_arr, to_update_arr, sizeof(T) * arrs_len, cudaMemcpyDeviceToHost);

	t* host_compare_arr = new t[arrs_len];
	cudaMemcpy(host_compare_arr, compare_arr, sizeof(T) * arrs_len, cudaMemcpyDeviceToHost);
	
	for (size_t i = 0; i < arrs_len; i++)
		if (compare_func(host_compare_arr[i], second_arg))
			host_arr[i] += to_add;

	T* out = 0;
	cudaMalloc(&out, sizeof(T) * arrs_len);
	cudaMemcpy(out, host_arr, sizeof(T) * arrs_len, cudaMemcpyHostToDevice);

	delete[] host_arr;
	delete[] host_compare_arr;

	if (free_updated) cudaFree(to_update_arr);
	return out;
}

template<typename T>
__host__ void save_array(T *arr, size_t arr_len, FILE *file, int is_device_arr)
{
	if (!arr) return;
	T* host_arr = new T[arr_len];
	cudaMemcpyKind		memcpy_kind = cudaMemcpyHostToHost;
	if (is_device_arr)	memcpy_kind = cudaMemcpyDeviceToHost;

	cudaMemcpy(host_arr, arr, sizeof(T) * arr_len, memcpy_kind);
	fwrite(host_arr, sizeof(T), arr_len, file);

	delete[] host_arr;
}

template<typename T>
__host__ T* load_array(size_t elem_count, FILE *file, int output_to_device)
{
	T* host_arr = new T[elem_count];
#ifdef WIN32
	if (fread_s(host_arr, sizeof(T) * elem_count, sizeof(T), elem_count, file) != sizeof(T) * elem_count) throw;
#else
	if (fread(host_arr, sizeof(T), elem_count, file)) throw;
#endif
	if (!output_to_device) return host_arr;
	
	T* device_arr = 0;
	cudaMalloc(&device_arr, sizeof(T) * elem_count);
	cudaMemcpy(device_arr, host_arr, sizeof(T) * elem_count, cudaMemcpyHostToDevice);
	delete[] host_arr;
	return device_arr;
}

template<typename T>
__host__ T load_value(FILE *file)
{
	T out;
	memset(&out, 0, sizeof(T));

#ifdef WIN32
	if (fread_s(&out, sizeof(T), sizeof(T), 1, file) != sizeof(T)) throw;
#else
	if (fread(&out, sizeof(T), 1, file)) throw;
#endif
	return out;
}

template<typename T>
__host__ T* cuda_push_back(T *old, size_t old_len, T new_last, bool free_old)
{
	T* out = cuda_realloc(old, old_len, old_len + 1, free_old);

	T tmp = new_last;
	cudaMemcpy(out + old_len, &tmp, sizeof(T), cudaMemcpyHostToDevice);
	return out;
}

template<typename T>
__host__ T* cuda_append_array(T* old, size_t old_len, T* to_append, size_t to_append_len, bool free_old, bool is_to_append_at_host = false)
{
	T* out = cuda_realloc(old, old_len, old_len + to_append_len, free_old);

	cudaMemcpyKind memcpykind = cudaMemcpyDeviceToDevice;
	if (is_to_append_at_host) memcpykind = cudaMemcpyHostToDevice;
	cudaMemcpy(out + old_len, to_append, to_append_len * sizeof(T), memcpykind);
	return out;
}

// if insert_i is < 0, an append is made
template<typename T>
__host__ T* cuda_insert_zeros(T* old, size_t old_len, long insert_i, size_t insert_len, bool free_old)
{
	T* out = 0;
	size_t out_len = old_len + insert_len;
	size_t out_size = sizeof(T) * out_len;

	if (insert_i > old_len) return 0;

	cudaMalloc(&out, out_size);
	if (!out) return 0;
	cudaMemset(out, 0, out_size);
	
	if (insert_i == old_len) insert_i = -1;
	if (insert_i < 0)
	{
		cudaMemcpy(out, old, sizeof(T) * old_len, cudaMemcpyDeviceToDevice);
		if (free_old) cudaFree(old);
		return out;
	}

	cudaMemcpy(out, old, sizeof(T) * insert_i, cudaMemcpyDeviceToDevice);
	cudaMemcpy(out + insert_i + insert_len, old + insert_i, sizeof(T) * (old_len - insert_i), cudaMemcpyDeviceToDevice);
	if (free_old) cudaFree(old);
	return out;
}

// Generates values between 0 and 1, divides them by value_divider and transforms 50% of them if generate_negative_values is true
template<typename T, typename t>
void generate_random_values(T* out, size_t value_count, size_t start_i = 0, t value_divider = 1, bool generate_negative_values = false)
{
	if (!out)
		return;
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
#ifdef DETERMINISTIC
	curandSetPseudoRandomGeneratorSeed(generator, 13);
#else
	curandSetPseudoRandomGeneratorSeed(generator, get_arbitrary_number());
#endif
	float* arr = 0;
	cudaMalloc(&arr, sizeof(float) * value_count);
	curandGenerateUniform(generator, arr, value_count);
	if (generate_negative_values)
	{
		add_to_array<float, float> kernel(value_count / 32 + (value_count % 32 > 0), 32) (
			arr, value_count, -.5
		);
		cudaDeviceSynchronize();
		multiply_array<float, float> kernel(value_count / 32 + (value_count % 32 > 0), 32) (
			arr, value_count, 2
		);
		cudaDeviceSynchronize();
	}
	multiply_array<float, t> kernel(value_count / 32 + (value_count % 32 > 0), 32) (
		arr, value_count, 1.0 / value_divider
	);
	cudaDeviceSynchronize();


	logical_copy<T, float> kernel(value_count / 32 + (value_count % 32 > 0), 32) ((out) + start_i, value_count, arr, value_count);
	cudaDeviceSynchronize();

	curandDestroyGenerator(generator);
	cudaDeviceSynchronize();
	cudaFree(arr);
}
