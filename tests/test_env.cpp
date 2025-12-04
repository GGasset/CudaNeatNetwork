
#include "test_env.h"
#include <cstdlib>

void test_env::initialize_env(size_t env, bool init_agent_pos)
{
	size_t board_len = board_size * board_size;
	size_t agent_pos = std::get<1>(target_agent_pos[env]);
	if (init_agent_pos)
		agent_pos = rand() % board_len;
	size_t target_pos = rand() % (board_len - 1);
	target_pos += target_pos >= agent_pos;
	target_agent_pos[env] = {target_pos, agent_pos};
	execution_n[env] = 0;
}

test_env::test_env(size_t _nenvs)
{
	nenvs = _nenvs;
	target_agent_pos.resize(nenvs, {0,0});
	execution_n.resize(_nenvs, 0);
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
		for (size_t i = 0; i < 4 && !selected_action; i++)
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
		throw;
	}
	size_t target_pos = std::get<0>(target_agent_pos[env_i]);
	data_t reward = 0;

	if (agent_pos == target_pos)
	{
		reward = 1;
		initialize_env(env_i, false);
		execution_n[env_i] = 0;
	}
	else
	{
		if (execution_n[env_i] > timeout)
		{
			initialize_env(env_i);

			int x = agent_pos % board_size;
			int y = agent_pos / board_size;

			int target_pos_x = target_pos % board_size;
			int target_pos_y = target_pos / board_size;

			size_t distance = abs(target_pos_x - x) * 2 + abs(target_pos_y - y) * 2;
			reward = -1 + (1 - distance / (data_t)(board_size * board_size));
			return {reward, true};
		}
		execution_n[env_i]++;
	}

	target_agent_pos[env_i] = { agent_pos, std::get<0>(target_agent_pos[env_i]) };
	return {reward, 0};
}
