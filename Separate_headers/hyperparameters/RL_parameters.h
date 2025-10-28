
#pragma once

#include "gradient_parameters.h"
#include <cstddef>

typedef struct GAE_hyperparameters
{
	gradient_hyperparameters value_function;

	// Discount factor, ranges [0, 1]
	// If its 1, all rewards in the future will be as important as a present one
	// If its 0, only current reward will count for the advantages
	data_t gamma = .995;

	// ranges [0, 1]
	// Should be close to 1
	data_t lambda = .98;

	// Number of times value function will be trained
	size_t training_steps = 15;

	// If false just subtracts value function predictions from empirical return to calculate advantage
	bool use_GAE = false;
} GAE_hyperparameters;

typedef struct PPO_hyperparameters
{
	gradient_hyperparameters policy;
	GAE_hyperparameters GAE;

	// Max training steps per PPO_train call to policy
	// May be overriden by early stopping
	size_t max_training_steps = 15;

	// Number of minibatches per training
	// If value or policy network are recurrent
	// mini_batch size % vec environment count MUST be 0
	size_t mini_batch_size = 4;

	// Environment frames before training is calculated and applied
	size_t steps_before_training = 30;

	// number of (independent) vector environments
	// if the execution count of an environment reach steps_before_training 
	// without the other environments having reached steps_before_training, an exception will be thrown
	size_t vecenvironment_count = 1;

	// Prevents each output from deviating from its initial output if it surpasses this ratio
	// Nullifies the gradient of the output
	// Typically from .1 to .3
	data_t clip_ratio = .2;

	// Used for early stopping
	// 0.01 or 0.05
	data_t max_kl_divergence_threshold = 0.05;
} PPO_hyperparameters;

