#pragma once
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <stdio.h>

template<typename t>
t h_max(t a, t b)
{
	return a * (a >= b) + b * (a < b);
}

template<typename t>
t h_min(t a, t b)
{
	return a * (a <= b) + b * (a > b);
}

unsigned long long get_arbitrary_number();

float get_random_float();

float Xavier_uniform_initialization_scale_factor(size_t n_inputs, size_t n_outputs);
