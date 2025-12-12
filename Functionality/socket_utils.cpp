#include "socket_utils.h"

socket_parser::~socket_parser()
{
	for (size_t i = 0; i < gathered_pointers.size(); i++) delete[] gathered_pointers[i];
	gathered_pointers.clear();
}

void* socket_parser::expected_read_arr(long expected_len)
{
	CHECK_ERRS();

	size_t len = read_var<size_t>(params);
	if (expected_len >= 0 && len != expected_len) HANDLE_ERR();
	if (*params.buff_pos + len > params.buff_len) HANDLE_ERR();

	char *out = new char[len + 1];
	if (!out) HANDLE_ERR();
	out[len] = 0;
	for (size_t i = 0; i < len; i++) out[i] = read_var<char>();
	gathered_pointers.push_back(out);
	return out;
}

std::tuple<void *, size_t> socket_parser::read_arr()
{
	
}
