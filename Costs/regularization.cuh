
#pragma once

#include "cuda_functionality.cuh"
#include "gradient_parameters.h"

class value_normalizer
{
private:
    size_t n = 0;
    data_t sum = 0;
    data_t sum_of_squares = 0;

    data_t mean = 0;
    data_t std = 0;

    void update_mean_std();

public:
    // Does not affect future normalization
    data_t normalize_val(data_t v);

    // Returns normalized vals, counts to future normalization
    data_t *incoming_vals(data_t *incoming_vals, size_t n_vals, bool are_vals_on_host, arr_location _return_location);
    // Returns normalized val, counts to future normalization
    data_t incoming_val(data_t v);
};

/// <summary>
/// Adds entropy regularization to the loss, which encourages exploration and prevents suboptimal convergence
/// </summary>
/// <param name="en_coefficient">Sets how much strenght the regularization has, if 0 the function will not try to do anyting, normally less than 0.01</param>
__host__ void entropy_regularization(
	size_t t_count, size_t neuron_count, size_t output_len,
	data_t* costs, data_t *activations, size_t last_layer_activations_start,
	entropy_bonus_hyperparameters hyperparameters
);

__host__ void global_gradient_clip(data_t *gradients, size_t gradient_count, gradient_hyperparameters hyperparameters);
