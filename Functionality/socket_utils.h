#pragma once

#include "stddef.h"
#include "data_type.h"

#include <tuple>
#include <vector>

#define HANDLE_ERR() {*params.err = true; return;}
#define CHECK_ERRS() if (!params.err) throw; if (*params.err) return; if (!params.buff || !params.buff_pos) HANDLE_ERR();

struct buff_params
{
	void	*buff;
	size_t	*buff_pos;
	size_t	buff_len;
	bool	*err;
};

class socket_parser 
{
private:
	buff_params params;
	std::vector<void *> gathered_pointers;

public:
	~socket_parser()

	template<typename T>
	T read_var(buff_params params)
	{
		CHECK_ERRS();
	
		if (*params.buff_pos + sizeof(T) > params.buff_len) HANDLE_ERR();
		T out = *(T*)((char *)params.buff + *params.buff_pos);
		*params.buff_pos += sizeof(T);
		return out;
	}
	
	void* expected_read_arr(buff_params params, long expected_len);
	std::tuple<void *, size_t> read_arr(buff_params params);
};

