
#pragma once

#include "data_type.h"

// Entropy regulariztion, adds a bonus term to the loss that moves values towards + - .367
//	-.367 is added for compatibility with negative activation functions such as tanh
// Encourages exploration and prevents suboptimal convergence
typedef struct
{
	bool active = false;

	// Sets the strength of the regularization, normally 0.01 or less
	data_t entropy_coefficient = .01;
} entropy_bonus_hyperparameters;

typedef struct
{
	entropy_bonus_hyperparameters entropy_bonus;
} regularization_hyperparameters;

typedef struct gradient_hyperparameters
{
	data_t learning_rate = .01;
	data_t gradient_clip = 1;
	float  dropout_rate = .2;

	regularization_hyperparameters regularization;
} gradient_hyperparameters;

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
} GAE_hyperparameters;

typedef struct PPO_hyperparameters
{
	gradient_hyperparameters policy;
	GAE_hyperparameters GAE;

	// Max training steps per PPO_train call to policy
	// May be overriden by early stopping
	size_t max_training_steps = 15;

	// Prevents each output from deviating from its initial output if it surpasses this ratio
	// Nullifies the gradient of the output
	// Typically from .1 to .3
	data_t clip_ratio = .2;

	// Used for early stopping
	// 0.01 or 0.05
	data_t max_kl_divergence_threshold = 0.05;
} PPO_hyperparameters;
