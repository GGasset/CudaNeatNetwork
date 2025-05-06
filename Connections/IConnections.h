#ifndef ICONNECTIONS_H
#define ICONNECTIONS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>

#include <stdlib.h>
#include <stdio.h>

#include "data_type.h"
#include "kernel_macros.h"
#include "NN_enums.h"
#include "linear_functions.cuh"
#include "connection_gradients.cuh"
#include "vector"
#include "evolution.h"
#include "cuda_functionality.cuh"
#include "functionality.h"

class IConnections
{
public:
	ConnectionTypes connection_type = ConnectionTypes::last_connection_entry;

	/// <summary>
	/// Device Array
	/// </summary>
	field_t* weights = 0;
	/// <summary>
	/// Device Array
	/// </summary>
	field_t* biases = 0;
	size_t neuron_count = 0;
	size_t connection_count = 0;
	unsigned char contains_irregular_connections = false;

	template<typename T, typename t>
	static void generate_random_values(T** pointer, size_t value_count, size_t start_i = 0, t value_divider = 1)
	{
		if (!pointer || !*pointer)
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
		multiply_array kernel(value_count / 32 + (value_count % 32 > 0), 32) (
			arr, value_count, 1.0 / value_divider
			);
		cudaDeviceSynchronize();

		logical_copy kernel(value_count / 32 + (value_count % 32 > 0), 32) ((*pointer) + start_i, value_count, arr, value_count);

		curandDestroyGenerator(generator);
		cudaDeviceSynchronize();
		cudaFree(arr);
	}

	virtual void linear_function(
		size_t activations_start, data_t* activations,
		data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
		) = 0;

	virtual void calculate_derivative(
		size_t activations_start, data_t* activations,
		size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
	) = 0;

	virtual void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		data_t* costs, size_t costs_start
	) = 0;

	virtual void subtract_gradients(
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		data_t learning_rate, data_t gradient_clip
	) = 0;

	virtual size_t get_connection_count_at(size_t neuron_i) = 0;

	virtual void mutate_fields(evolution_metadata evolution_values);
	virtual void add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections);
	virtual void adjust_to_added_neuron(size_t added_neuron_i, float connection_probability, std::vector<size_t>* added_connections_neuron_i);
	virtual void remove_neuron(size_t layer_neuron_i);
	virtual void adjust_to_removed_neuron(size_t neuron_i, std::vector<size_t>* removed_connections_neuron_i);
	
	virtual IConnections* connections_specific_clone() = 0;
	void IConnections_clone(IConnections* base);

	virtual void specific_save(FILE* file) = 0;
	void save(FILE* file);

	virtual void load(FILE* file) = 0; 
	void load_neuron_metadata(FILE* file);
	void load_IConnections_data(FILE* file);

	void deallocate();
	virtual void specific_deallocate();
};

#endif