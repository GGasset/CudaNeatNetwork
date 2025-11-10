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

template<typename t>
void vec_shuffle_inplace(std::vector<t> &inp)
{
	size_t rand_val = rand() % (inp.size() - 1);
	for (size_t i = 0; i < inp.size(); i++, rand_val = rand() % (inp.size() - 1)) inp[i] = inp[rand_val + (rand_val >= i)];
}

unsigned long long get_arbitrary_number();

float get_random_float();

float Xavier_uniform_initialization_scale_factor(size_t n_inputs, size_t n_outputs);
