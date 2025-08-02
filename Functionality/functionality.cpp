#include "functionality.h"

unsigned long long get_arbitrary_number()
{
	return (unsigned long long)clock();
}

float get_random_float()
{
    return rand() % 10000 / 10000.0;
}

float Xavier_uniform_initialization_scale_factor(size_t n_inputs, size_t n_outputs)
{
    return sqrtf(6.0 / (n_inputs + n_outputs));
}
