
#include "test_env.h"
#include <cstdlib>
#include <cstring>
#include <utility>
#include <iostream>

void test_env::add_to_last_episode_lens(size_t episode_len)
{
	if (last_episode_lens.size() >= n_last_episode_lens)
		last_episode_lens.erase(last_episode_lens.begin());
	last_episode_lens.push_back(episode_len);
}

std::tuple<data_t, bool> test_env::end_of_episode(data_t reward, size_t env_i)
{
	add_to_last_episode_lens(execution_n[env_i]);
	initialize_env(env_i);
	return {reward, 1};
}

void test_env::initialize_env(size_t env, bool init_agent_pos)
{
	size_t board_len = board_size * board_size;
	size_t agent_pos = std::get<1>(target_agent_pos[env]);
	size_t target_pos = 0;
	if (!easy_mode)
	{
		if (init_agent_pos)
			agent_pos = rand() % board_len;
		target_pos = rand() % (board_len - 1);
		target_pos += target_pos >= agent_pos;
	}
	else
	{
		agent_pos = 0;
		target_pos = 1;
	}
	target_agent_pos[env] = {target_pos, agent_pos};
	execution_n[env] = 0;
}

test_env::test_env(size_t _nenvs, bool easy_mode)
{
	this->easy_mode = easy_mode;

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

std::tuple<data_t *, size_t> test_env::get_all_observations()
{
	size_t total_observation_count = get_observation_count() * nenvs;
    data_t *out_arr = new data_t[total_observation_count];

	size_t actual_observations = 0;
	for (size_t i = 0; i < nenvs; i++)
	{
		size_t observations_start_i = i * get_observation_count();

		std::vector<data_t> obs = get_observations(i);
		memcpy(out_arr + observations_start_i, obs.data(), sizeof(data_t) * obs.size());

		actual_observations += obs.size();
	}
	if (actual_observations != total_observation_count) { delete[] out_arr; throw; };

	return {out_arr, total_observation_count};
}

std::tuple<data_t, bool> test_env::step(data_t *actions_probs, size_t env_i)
{
	size_t x = std::get<1>(target_agent_pos[env_i]) % board_size;
	size_t y = std::get<1>(target_agent_pos[env_i]) / board_size;


	if (!actions_probs) 
		throw;

	size_t selected_action = 0;
	data_t r = get_random_float();
	data_t cumulative_probs = 0;
	{
		bool contains_nan = 0;
		for (size_t i = 0; i < 4 && !selected_action; i++)
		{
			contains_nan = contains_nan || actions_probs[i] != actions_probs[i];
			cumulative_probs += actions_probs[i];
			if (r <= cumulative_probs)
				selected_action = i + 1;
		}
		if (!selected_action && !contains_nan)
			selected_action = 4;
	}

	size_t agent_pos = std::get<1>(target_agent_pos[env_i]);
	switch (selected_action)
	{
	case 1:
		if (x == 0) 
			return end_of_episode(-1, env_i);
		agent_pos--;
		break;
	case 2:
		if (x == board_size - 1) 
			return end_of_episode(-1, env_i);
		agent_pos++;
		break;
	case 3:
		if (y == 0) 
			return end_of_episode(-1, env_i);
		agent_pos -= board_size;
		break;
	case 4:
		if (y == board_size - 1) 
			return end_of_episode(-1, env_i);
		agent_pos += board_size;
		break;

	default:
		printf("Invalid action %li  %.2f  %.2f\n", selected_action, r, cumulative_probs);
		throw;
	}
	size_t target_pos = std::get<0>(target_agent_pos[env_i]);

	if (agent_pos == target_pos)
	{
		return end_of_episode(1, env_i);
	}
	else
	{
		if (execution_n[env_i] > timeout)
		{
			int x = agent_pos % board_size;
			int y = agent_pos / board_size;

			int target_pos_x = target_pos % board_size;
			int target_pos_y = target_pos / board_size;

			size_t distance = abs(target_pos_x - x) + abs(target_pos_y - y);
			data_t reward = -1 + 2 * (1 - distance / (data_t)(board_size * board_size));
			return end_of_episode(reward, env_i);
		}
		execution_n[env_i]++;
	}

	target_agent_pos[env_i] = { agent_pos, std::get<0>(target_agent_pos[env_i]) };
	return {0, 0};
}

std::tuple<data_t *, size_t, std::vector<bool>> test_env::step(data_t *all_action_probs)
{
	data_t *rewards = new data_t[nenvs];
	std::vector<bool> EOPs;
	EOPs.resize(nenvs);

    for (size_t i = 0; i < nenvs; i++)
	{
		size_t action_probs_start_i = get_action_count() * i;

		auto [reward, EOP] = step(all_action_probs + action_probs_start_i, i);

		rewards[i] = reward;
		EOPs[i] = EOP;
	}
	return {rewards, nenvs, EOPs};
}

data_t test_env::get_mean_episode_len()
{
	if (!last_episode_lens.size())
		return 0;
	data_t sum = 0;
	for (size_t i = 0; i < last_episode_lens.size(); i++)
	{
		sum += last_episode_lens[i];
	}
	return sum /= last_episode_lens.size();
}
