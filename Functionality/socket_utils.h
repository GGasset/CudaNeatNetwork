#pragma once

#include "stddef.h"
#include "data_type.h"

#include <type_traits>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <utility>

template <std::size_t I = 0, typename Tuple, typename Value>
void set_tuple_value(Tuple& t, std::size_t index, Value&& value) {
    if constexpr (I == std::tuple_size_v<Tuple>) {
        throw std::out_of_range("tuple index out of range");
    } else {
        if (I == index) {
            std::get<I>(t) = std::forward<Value>(value);
        } else {
            set_tuple_value<I + 1>(t, index, std::forward<Value>(value));
        }
    }
}


struct buff_params
{
	void	*buff;
	size_t	buff_len;
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
	size_t buff_pos = 0;

	std::tuple<function_param_types> parameters;
	std::vector<void*> pointers_to_delete;
	size_t parsing_param_i = 0;
	bool must_parse_length = false;
	size_t last_parsed_len = 0;

	template<typename T>
	T read_val()
	{
		if (buff_pos + sizeof(T) > params.buff_len) throw;
		T out = *(T*)((char*)params.buff + buff_pos);
		buff_pos += sizeof(T);
		return out;
	}

	template<typename T>
	std::tuple<T*,size_t> read_arr()
	{
		size_t len = read_val<size_t>();
		if (buff_pos + sizeof(T) * len > params.buff_len) throw;
		T *out = new T[len];
		try
		{
			for (size_t i = 0; i < len; i++) out[i] = read_val<T>();
		}
		catch { delete[] out; throw; }

		pointers_to_delete.push_back(out);
		return {len, out};
	}

	// throws on error, then recommended to discard the buffer
	// For security reasons and accompanied with pain from the creator
// After a pointer is passed a size_t which will be used as parsed array len must be specified in template arguments
// Thus called functions must have an array len parameter after an array pointer
// An array is preceded and superceded (enveloped) in the buffer by its length in size_t
	template<typename param_type>
	void parse_parameter()
	{
		param_type parsed;

		if (std::is_pointer<param_type>::value)
		{
			if (must_parse_length) throw;
			
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
			
			must_parse_length = false;
			last_parsed_len = 0;
		}

		set_tuple_value(parameters, parsing_param_i, parsed);
		parsing_param_i++;
		return parsed;
	}

	parse_apply();
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
	// this function does support member functions
	// Uses lambda function to call the template function withing a object
	// Template member_function argument must be passed as &obj::foo
	template<typename object_type, typename member_function>
	auto operator()(object_type &obj)
	{
		(parse_parameter<func_param_types>(), ...);
		return std::apply([&](auto&&... args) { (obj.*member_function)(std::forward<decltype(args)>(args)...); }, parameters);
	}
};





