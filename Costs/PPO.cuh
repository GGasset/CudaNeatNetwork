#pragma once

#include "kernel_macros.h"
#include "data_type.h"
#include "NN_enums.h"

#include "RL_parameters.h"

#include "GAE.cuh"
#include "NN.h"

struct PPO_internal_memory
{
	std::vector<data_t *>	initial_states;
	std::vector<data_t *>	trajectory_inputs;
	std::vector<data_t *>	trajectory_outputs;
	std::vector<data_t *>	rewards;
	std::vector<size_t>		n_env_executions;
	size_t					n_env = 0;
};


// PPO execution, will train automatically after steps_before_training_steps
// save_this_for_me is recommended to be on the stack, unless there are multiple policy networks
// After calling this function and receiving the action by the return value, call add reward with save_this_for_me
data_t *PPO_execute_train(
	data_t *X,
	NN *value_function, NN *policy, PPO_hyperparameters hyperparameters,
	PPO_internal_memory *save_this_for_me, output_pointer_type output_kind
);

void add_reward(
	data_t reward, size_t env_i, PPO_internal_memory *save_this_for_me
);

// returns 0
bool free_PPO_data(PPO_internal_memory *mem);
