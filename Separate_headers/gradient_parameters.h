
#include "data_type.h"

typedef struct gradient_hyperparameters
{
	data_t learning_rate = .01;
	data_t gradient_clip = 50;
	float  dropout_rate = .2;
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
};

typedef struct PPO_hyperparameters
{
	gradient_hyperparameters policy;
	GAE_hyperparameters GAE;

	// Max training steps per PPO_train call to policy
	// May be overriden by early stopping
	size_t max_training_steps = 50;

	// Prevents each output from deviating from its initial output if it surpasses this ratio
	// Nullifies the gradient of the output
	// Typically from .1 to .3
	data_t clip_ratio = .2;

	// Used for early stopping
	// 0.01 or 0.05
	data_t max_kl_divergence_threshold = 0.05;
};
