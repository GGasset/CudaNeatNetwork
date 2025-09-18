
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
