#include "socket_utils.h"

socket_parser::~socket_parser()
{
	for (size_t i = 0; i < gathered_pointers.size(); i++) delete[] gathered_pointers[i];
	gathered_pointers.clear();
}

void* socket_parser::expected_read_arr(long expected_len)
{
	auto [out, len] = read_arr();
	if (expected_len >= 0 && len != expected_len) {*params.err = true; return 0;}
	return out;
}

std::tuple<void *, size_t> socket_parser::read_arr()
{
	CHECK_ERRS();

	size_t len = read_var<size_t>(params);
	if (*params.buff_pos + len > params.buff_len) {*params.err = true; return {0,0};}

	char *out = new char[len + 1];
	if (!out) HANDLE_ERR();
	out[len] = 0;
	for (size_t i = 0; i < len; i++) out[i] = read_var<char>();
	gathered_pointers.push_back(out);
	return {out, len};
}
