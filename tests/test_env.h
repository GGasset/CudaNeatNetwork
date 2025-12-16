
#pragma once
#include <cstddef>
#include <vector>

#include "data_type.h"
#include "functionality.h"


// test_env is a simple RL grid pathfinding prototype to test PPO
class test_env
{
private:
	const size_t board_size = 3;
	const size_t timeout = board_size * board_size * 4;

	std::vector<std::tuple<size_t, size_t>> target_agent_pos;
	std::vector<size_t> execution_n;
	size_t nenvs;

	std::vector<size_t> last_episode_lens;
	size_t n_last_episode_lens = 800;
	void add_to_last_episode_lens(size_t episode_len);

	std::tuple<data_t, bool> end_of_episode(data_t reward, size_t env_i);

	void initialize_env(size_t env, bool init_agent_pos = true);
public:
	test_env(size_t _n_envs);

	std::vector<data_t> get_observations(size_t env_i);

	// Returns reward and wheter the episode finished
	// Actions_probs must be a host arr
	std::tuple<data_t, bool> step(data_t *actions_probs, size_t env_i);

	data_t get_mean_episode_len();
	inline size_t get_observation_count() {return 4; }
	inline size_t get_action_count() { return 4; }
	inline size_t get_last_episode_lens_len() {return last_episode_lens.size();}
};
