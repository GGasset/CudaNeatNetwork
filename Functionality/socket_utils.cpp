#include "socket_utils.h"

parse_apply::parse_apply(void *buff, size_t buff_len)
{
	parameters.buff = buff;
	parameters.buff_len = buff_len;
}

parse_apply::~parse_apply()
{
	for (size_t i = 0; i < pointer_to_delete.size(); i++) {delete[] pointers_to_delete[i]; pointers_to_delete[i] = 0;}
	pointers_to_delete.clear();
}
