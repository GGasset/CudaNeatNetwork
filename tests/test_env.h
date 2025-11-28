
#pragma once
#include <cstddef>
#include <vector>

#include "data_type.h"
#include "functionality.h"


// test_env is a simple RL grid pathfinding prototype to test PPO
class test_env
{
private:
	static const size_t board_size = 3;

	std::vector<std::tuple<size_t, size_t>> target_agent_pos;
	size_t nenvs;

	void initialize_env(size_t env);
public:
	test_env(size_t _n_envs);

	std::vector<data_t> get_observations(size_t env_i);

	// Returns reward and wheter the episode finished
	// Actions_probs must be a host arr
	std::tuple<data_t, bool> step(data_t *actions_probs, size_t env_i);


};
