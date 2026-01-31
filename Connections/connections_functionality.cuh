
#pragma once

#include "cuda_functionality.cuh"
#include "nn_lens.h"

// Will do element wise multiply, when the smaller array ends, it starts from the beginning, stops when the bigger array ends
// If lengths aren't in order, no operation is done
// out must be of len big_len
// out may equal to big, not small
template<typename Ta, typename Tb, typename Tc>
__global__ void repetitive_element_wise_multiply (
	Ta *big, size_t big_len, Tb *small, size_t small_len, Tc *out
)
{
	size_t tid = get_tid();
	if (tid >= big_len || big_len < small_len) return;

	size_t small_i = tid % small_len;
	out[tid] = big[tid] * small[small_i];
}

// Will sum 2 arrays, when the smaller array ends, it starts from the beginning, stops when the bigger array ends
// If lengths aren't in order, no operation is done
// out must be of len big_len
// out may equal to big
template<typename Ta, typename Tb, typename Tc>
__global__ void repetitive_sum (
	Ta *big, size_t big_len, Tb *small, size_t small_len, Tc *out
)
{
	size_t tid = get_tid();
	if (tid >= big_len || big_len < small_len) return;

	size_t small_i = tid % small_len;
	out[tid] = big[tid] + small[small_i];
}

__global__ void extract_activations_dense(
	size_t t_count, data_t *activations, size_t neuron_count, size_t layer_neuron_count,
	data_t *out_arr, size_t previous_layer_activations_start, size_t previous_layer_length,
	size_t gaps_between_usable_arrays_t_count
);

// doesn't just extract activations as the NEAT irregularities would make processing more costly, as my multi PRAM needs normalized size arrays
// out arr must be of length max_connection_count_at_layer * t_count, use n_thread(max_conns * layer_neuron_count * t_count)
// if a neuron doesn't contain max_connection_count connections, the empty spaces are set to 0, so PRAM add is not affected by them
__global__ void get_pondered_activations_neat(
	size_t t_count, data_t *activations, size_t neuron_count, size_t layer_neuron_count, size_t connection_count,
	data_t *out_arr, size_t *connection_points, size_t *connection_neuron_i, data_t *weights, size_t max_connection_count_at_layer,
	size_t gaps_between_usable_arrays_t_count
);

__global__ void insert_execution_values(
	size_t t_count, size_t nn_execution_value_count, size_t to_insert_layer_neuron_count,
	size_t layer_execution_values_start, size_t execution_values_per_neuron, size_t neuron_execution_values_i, 
	size_t gaps_between_usable_arrays_t_count,
	data_t *to_insert, data_t *execution_values
);
