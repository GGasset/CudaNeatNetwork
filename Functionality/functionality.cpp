#include "functionality.h"

float get_random_float()
{
    return rand() % 10000 / 10000.0;
}

unsigned long long get_arbitrary_number()
{
	return (unsigned long long)clock();
}