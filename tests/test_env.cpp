
#include "test_env.h"
#include <cstdlib>

void test_env::initialize_env(size_t env)
{
	size_t board_len = board_size * board_size;
	size_t agent_pos = rand() % board_len;
	size_t target_pos = rand() % (board_len - 1);
	target_pos += target_pos >= agent_pos;
	target_agent_pos[i] = {target_pos, agent_pos};
}

test_env::test_env(size_t _nenvs)
{
	_nenvs = nenvs;
	target_agent_pos.resize(nenvs, {0,0});
	for (size_t i = 0; i < nenvs; i++)
		initialize_env(i);
}

std::vector<data_t> test_env::get_observations(size_t env_i)
{
	if (env_i >= nenvs) throw;

	std::vector<data_t> out;

	size_t target_pos = std::get<0>(target_agent_pos[env_i]);
	size_t target_pos_components[2] {};
	target_pos_components[0] = target_pos % board_size;
	target_pos_components[1] = target_pos / board_size;

	size_t agent_pos = std::get<1>(target_agent_pos[env_i]);
	size_t agent_pos_components[2] {};
	agent_pos_components[0] = target_pos % board_size;
	agent_pos_components[1] = target_pos / board_size;

	// Target direction
	data_t tmp = target_pos_components[0] - agent_pos_components[0];
	tmp = (tmp != 0) * (1 - 2 * (tmp < 0));
	out.push_back(tmp);

	tmp = target_pos_components[1] - agent_pos_components[1];
	tmp = (tmp != 0) * (1 - 2 * (tmp < 0));
	out.push_back(tmp);

	// Board edge
	out.push_back((agent_pos_components[0] == board_size - 1) - (agent_pos_components[0] == 0));
	out.push_back((agent_pos_components[1] == board_size - 1) - (agent_pos_components[1] == 0));

	return out;
}

std::tuple<data_t, bool> test_env::step(data_t *actions_probs, size_t env_i)
{
	size_t x = std::get<1>(target_agent_pos[env_i]) % board_size;
	size_t y = std::get<1>(target_agent_pos[env_i]) / board_size;

	if (!actions_probs) 
		throw;

	size_t selected_action = 0;
	{
		data_t r = get_random_float();
		data_t cumulative_probs = 0;
		for (size_t i = 0; i < 4; i++)
		{
			cumulative_probs += actions_probs[i];
			if (r <= cumulative_probs)
				selected_action = i + 1;
		}
	}

	size_t agent_pos = std::get<1>(target_agent_pos[env_i]);
	switch (selected_action)
	{
	case 1:
		if (x == 0) 
		{
			initialize_env(env_i);
			return {-1, 1};
		}
		agent_pos--;
		break;
	case 2:
		if (x == board_size - 1) 
		{
			initialize_env(env_i);
			return {-1, 1};
		}
		agent_pos++;
		break;
	case 3:
		if (y == 0) 
		{
			initialize_env(env_i);
			return {-1, 1};
		}
		agent_pos -= board_size;
		break;
	case 4:
		if (y == board_size - 1) 
		{
			initialize_env(env_i);
			return {-1, 1};
		}
		agent_pos += board_size;
		break;

	default:
		return {0, 0};
	}
}
