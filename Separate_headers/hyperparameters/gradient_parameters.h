
#pragma once

#include "data_type.h"
#include "loss_hyperparameters.h"


typedef struct Adam_hyperparameters
{
	bool active = true;

	data_t beta_1 = .9;
	data_t beta_2 = .999;
	data_t epsilon = 1e-8;
} Adam_hyperparameters;

// Mix of L1 and L2, applied independently of the gradient
typedef struct ElasticNet_hyperparameters
{
	bool active = true;

	// Regularization coefficient - Usually in the range of 1E-5 and 1
	data_t lambda = 1E-4;
	
	// L1-L2 ratio, alpha=0 is L2, alpha=1 is L1
	data_t alpha = .1;
} ElasticNet_hyperparameters;

typedef struct optimizer_hyperparameters
{
	// Adaptive moment estimation, improves efficiency of learning with momentum
	Adam_hyperparameters adam;

	// Mix of L1 and L2, applied independently of the gradient
	// Used to prevent overfitting by leading weights to 0
	// L1 can lead a weight to 0, it does feature selection and is less sensitive to outliers
	// L2 never leads a weight to 0, it leads outlier weights to 0 more strongly than the ones closer to 0
	ElasticNet_hyperparameters L_regularization;
} optimizer_hyperparameters;


typedef struct gradient_hyperparameters
{
	data_t learning_rate = .001;
	// Rescales the L2 norm of the timestep gradients to the value
	// Set to 0 to disable
	data_t global_gradient_clip = .5;
	data_t gradient_clip = 1;
	float  dropout_rate = .2;

	regularization_hyperparameters	regularization;
	optimizer_hyperparameters		optimization;
} gradient_hyperparameters;
