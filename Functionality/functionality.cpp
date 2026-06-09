#include "functionality.h"
#include <time.h>
#include <cstdint>

uint64_t xorshift64() 
{
#ifdef DETERMINISTIC
	static uint64_t x = 13;
#else
	static uint64_t x = get_arbitrary_number();
#endif

	x ^= x << 7;
	x ^= x >> 9;
	return x;
}

unsigned long long get_arbitrary_number()
{
#ifdef DETERMINISTIC
    return xorshift64();
#else
    return ((int)time(NULL)) +
            (unsigned long long)clock() +
            rand();
#endif
}

float get_random_float()
{
    return get_arbitrary_number() % 10000 / 10000.0;
}

float Xavier_uniform_initialization_scale_factor(size_t n_inputs, size_t n_outputs)
{
    return sqrtf(6.0 / (n_inputs + n_outputs));
}
