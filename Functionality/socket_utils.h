#pragma once

#include "stddef.h"
#include "data_type.h"

#include <type_traits>
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
	socket_parser(void *buff, size_t buff_len);
	~socket_parser()

	template<typename T>
	T read_var()
	{
		CHECK_ERRS();
	
		if (*params.buff_pos + sizeof(T) > params.buff_len) HANDLE_ERR();
		T out = *(T*)((char *)params.buff + *params.buff_pos);
		*params.buff_pos += sizeof(T);
		return out;
	}
	
	void* expected_read_arr(long expected_len);
	std::tuple<void *, size_t> read_arr();
};

// For security reasons and accompanied with pain from the creator
// After a pointer is passed a size_t which will be used as parsed array len must be specified in template arguments
// Thus called functions must have an array len parameter after an array pointer
// An array is preceded (to check read length) and superceded (so it can be checked by the array processing function) (enveloped) in the buffer by its length in size_t
// The preceding size_t must not be passed as func_param_types, as it will not be passed to the function, its for security purposes only
template<typename... func_param_types>
class parse_apply
{
private:
	buff_params params;
	std::tuple<function_param_types> parameters;
	std::vector<void*> pointers_to_delete;
	size_t parsing_param_i = 0;
	bool must_parse_length = false;
	size_t last_parsed_len = 0;

	template<typename T>
	T read_val()
	{
		
	}

	template<typename T>
	std::tuple<T*,size_t> read_arr()
	{
		
	}

	// throws on error, then recommended to discard the buffer
	// For security reasons and accompanied with pain from the creator
// After a pointer is passed a size_t which will be used as parsed array len must be specified in template arguments
// Thus called functions must have an array len parameter after an array pointer
// An array is preceded and superceded (enveloped) in the buffer by its length in size_t
	template<typename param_type>
	void parse_parameter()
	{
		param_type parsed = 0;

		if (std::is_pointer<param_type>::value)
		{
			auto [parsed, len] = read_arr<std::remove_pointer<param_type>::type>();
			last_parsed_len = len;
			must_parse_length = true;
		}
		else
		{
			if (!std::is_same_v<param_type, size_t>)
				assert(!must_parse_length);
	
			parsed = read_val<param_type>();
			if (must_parse_length && last_parsed_len != parsed) throw;
			
			parsing_param_i++;
			must_parse_length = false;
			last_parsed_len = 0;
		}
		return parsed;
	}

public:
	parse_apply(void *buff, size_t expected_buff_len);
	~parse_apply();

	// throws on error, then recommended to discard the buffer
	// this function does not support member functions
	template<typename non_member_function> // May also return based on a template type (note for future development)
	auto operator()()
	{
		(parse_parameter<func_param_types>(), ...);
		return std::apply(non_member_function, parameters);
	}

	// throws on error, then recommended to discard the buffer
	// this function does not support member functions
	// Uses lambda function to call the template function withing a object
	template<typename object_type, typename member_function>
	auto operator()(object_type &obj)
	{
		(parse_parameter<func_param_types>(), ...);
		return std::apply([&](auto&&... args) { (obj.*member_function)(std::forward<decltype(args)>(args)...); }, parameters);
	}
};





