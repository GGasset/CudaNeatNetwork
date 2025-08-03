
#pragma once

#include "cuda_functionality.cuh"
#include "gradient_parameters.h"

/// <summary>
/// Adds entropy regularization to the loss, which encourages exploration and prevents suboptimal convergence
/// </summary>
/// <param name="en_coefficient">Sets how much strenght the regularization has, if 0 the function will not try to do anyting, normally less than 0.01</param>
__host__ void entropy_regularization( 
	size_t t_count, size_t neuron_count, size_t output_len,
	data_t* costs, data_t *activations, size_t last_layer_activations_start,
	entropy_bonus_hyperparameters hyperparameters
);

