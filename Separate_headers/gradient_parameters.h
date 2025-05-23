
#include "data_type.h"

typedef struct gradient_hyperparameters
{
	data_t learning_rate = .01;
	data_t gradient_clip = 50;
	float  dropout_rate = .2;
} gradient_hyperparameters;
