
#pragma once

#include "cuda_functionality.cuh"
#include "gradient_parameters.h"

class reward_normalization_data
{
private:
    data_t mean = 0;
    data_t std = 0;

public:
    // Does not affect future normalization
    data_t normalize_reward(data_t r);

    // Returns normalized rewards, counts to future normalization
    data_t *incoming_rewards(data_t *incoming_rewards, bool are_rewards_on_host, arr_location _return_location);
    // Returns normalized reward, counts to future normalization
    data_t incoming_reward(data_t r);
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
