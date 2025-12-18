#ifndef NN_DEFINITIONS
#define NN_DEFINITIONS

#include "NN.h"

size_t NN::get_input_length()
{
	return input_length; 
}

size_t NN::get_neuron_count()
{
	return neuron_count;
}

size_t NN::get_output_length()
{
	return output_length;
}

size_t NN::get_output_activations_start()
{
	return *output_activations_start;
}

size_t NN::get_gradient_count_per_t()
{
	return gradient_count;
}

bool NN::is_recurrent()
{
	return contains_recurrent_layers;
}

NN::NN(
	ILayer** layers, size_t input_length, size_t layer_count,
	initialization_parameters weight,
	initialization_parameters bias,
	initialization_parameters layer_weight
)
{
	weight_init = weight;
	bias_init = bias;
	layer_weights_init = layer_weight;

	this->layers = layers;
	this->input_length = input_length;
	this->layer_count = layer_count;
	set_fields();
}

NN::NN()
{
}

NN::~NN()
{
	deallocate();
}

void NN::set_fields()
{
	output_length = layers[layer_count - 1]->get_neuron_count();

	size_t neuron_count = input_length;
	size_t execution_value_count = 0;
	size_t derivative_count = 0;
	size_t gradient_count = 0;
	contains_recurrent_layers = false;
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* layer = layers[i];
		
		contains_recurrent_layers = contains_recurrent_layers || layer->is_recurrent;

		layer->layer_activations_start = neuron_count;
		neuron_count += layer->get_neuron_count();

		layer->execution_values_layer_start = execution_value_count;
		execution_value_count += layer->execution_values_per_neuron * layer->get_neuron_count();

		layer->layer_derivatives_start = derivative_count;
		derivative_count += layer->layer_derivative_count;

			layer->layer_gradients_start = gradient_count;
			gradient_count += layer->layer_gradient_count;
		}
		this->neuron_count = neuron_count;
	output_activations_start = &(layers[layer_count - 1]->layer_activations_start);
	this->execution_value_count = execution_value_count;
	this->derivative_count = derivative_count;
	this->gradient_count = gradient_count;
}

void NN::execute(data_t* input, data_t* execution_values, data_t* activations, size_t t, data_t* output_start_pointer, output_pointer_type output_type)
{
	cudaMemcpy(activations + t * neuron_count, input + input_length * t, sizeof(data_t) * input_length, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->execute(activations, neuron_count * t, execution_values, execution_value_count * t);
		cudaDeviceSynchronize();
	}
	if (output_type != no_output && output_start_pointer)
	{
		cudaMemcpyKind memcpy_kind = cudaMemcpyDeviceToHost;
		if (output_type == cuda_pointer_output) memcpy_kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy(output_start_pointer + output_length * t, activations + neuron_count * t + *output_activations_start, sizeof(data_t) * output_length, memcpy_kind);
		cudaDeviceSynchronize();
	}
}

void NN::set_up_execution_arrays(data_t** execution_values, data_t** activations, size_t t_count)
{
	cudaMalloc(execution_values, sizeof(data_t) * execution_value_count * t_count);
	cudaMalloc(activations, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();
	cudaMemset(*execution_values, 0, sizeof(data_t) * execution_value_count * t_count);
	cudaMemset(*activations, 0, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();
}

data_t* NN::batch_execute(data_t* input, size_t t_count, output_pointer_type output_type)
{
	if (output_type == no_output) return (0);

	data_t* execution_values = 0;
	data_t* activations = 0;
	set_up_execution_arrays(&execution_values, &activations, t_count);

	data_t* outputs = alloc_output(output_length * t_count, output_type);
	for (size_t t = 0; t < t_count; t++)
		execute(input, execution_values, activations, t, outputs, output_type);

	cudaFree(execution_values);
	cudaFree(activations);
	cudaDeviceSynchronize();
	return outputs;
}

data_t* NN::inference_execute(data_t* input, output_pointer_type output_type)
{
	return batch_execute(input, 1, output_type);
}

data_t NN::adjust_learning_rate(
	data_t learning_rate,
	data_t cost,
	LearningRateAdjusters adjuster,
	data_t max_learning_rate,
	data_t previous_cost
)
{
	data_t new_learning_rate = learning_rate;
	if (adjuster == LearningRateAdjusters::none) return new_learning_rate;
	if (previous_cost != 0 && cost != 0)
		switch (adjuster) {
			case LearningRateAdjusters::high_learning_high_learning_rate:
				{
					data_t learning = previous_cost / cost;
					new_learning_rate += learning;
				}
				break;
			case LearningRateAdjusters::high_learning_low_learning_rate:
				{
					data_t learning = previous_cost / cost;
					new_learning_rate -= learning;
					new_learning_rate = h_max((data_t)0, new_learning_rate);
				}
				break;
			default:
				break;
		}
	switch (adjuster) {
		case LearningRateAdjusters::cost_times_learning_rate:
			new_learning_rate = learning_rate * cost;
			break;
		default:
			break;
	}
	return h_min(new_learning_rate, max_learning_rate);

}

data_t NN::training_batch(
	size_t t_count,
	data_t* X,
	data_t* Y_hat,
	bool is_Y_hat_on_host_memory,
	size_t Y_hat_value_count,
	CostFunctions cost_function,
	data_t** Y,
	output_pointer_type output_type,
	gradient_hyperparameters hyperparameters
)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	training_execute(
		t_count,
		X,
		Y,
		output_type,
		&execution_values,
		&activations
	);
	return train(
		t_count,
		execution_values,
		activations,
		Y_hat,
		is_Y_hat_on_host_memory,
		Y_hat_value_count,
		cost_function,
		hyperparameters
	);
}

void NN::training_execute(
	size_t t_count,
	data_t* X,
	data_t** Y,
	output_pointer_type output_type,
	data_t** execution_values,
	data_t** activations,
	size_t arrays_t_length,
	std::vector<bool> *delete_mem
)
{
	data_t* prev_execution_values = 0;
	data_t* prev_activations = 0;
	if (arrays_t_length)
	{
		prev_execution_values = *execution_values;
		prev_activations = *activations;
	}
	set_up_execution_arrays(execution_values, activations, t_count + arrays_t_length);
	if (arrays_t_length)
	{
		cudaMemcpy(*execution_values, prev_execution_values, sizeof(data_t) * execution_value_count * arrays_t_length, cudaMemcpyDeviceToDevice);
		cudaMemcpy(*activations, prev_activations, sizeof(data_t) * neuron_count * arrays_t_length, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
		cudaFree(prev_execution_values);
		cudaFree(prev_activations);
	}


	data_t* out_Y = 0;
	if (Y)
	{
		out_Y = alloc_output(t_count * output_length, output_type);
		*Y = out_Y;
	}
	for (size_t t = 0; t < t_count; t++)
	{
		if (delete_mem && delete_mem->size() > t && delete_mem[0][t]) delete_memory();
		execute(X, (*execution_values) + execution_value_count * arrays_t_length, (*activations) + neuron_count * arrays_t_length, t, out_Y, output_type);
	}
}


data_t NN::train(
	size_t t_count,
	data_t* execution_values,
	data_t* activations,
	data_t* Y_hat,
	bool is_Y_hat_on_host_memory,
	size_t Y_hat_value_count,
	CostFunctions cost_function,
	gradient_hyperparameters hyperparameters
)
{
	data_t* costs = 0;
	cudaMalloc(&costs, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();

	cudaMemset(costs, 0, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();
	
	if (is_Y_hat_on_host_memory)
	{
		data_t* temp_Y_hat = 0;
		cudaMalloc(&temp_Y_hat, sizeof(data_t) * Y_hat_value_count);
		cudaMemcpy(temp_Y_hat, Y_hat, sizeof(data_t) * Y_hat_value_count, cudaMemcpyHostToDevice);
		Y_hat = temp_Y_hat;
	}
	data_t cost = calculate_output_costs(cost_function, t_count, Y_hat, activations, 0, costs, 0);
	cudaDeviceSynchronize();

	data_t* gradients = 0;
	backpropagate(
		t_count, costs, activations, execution_values, &gradients, hyperparameters
	);

	for (size_t t = 0; t < t_count; t++)
	{
		subtract_gradients(gradients, gradient_count * t, hyperparameters);
	}

	if (is_Y_hat_on_host_memory) cudaFree(Y_hat);
	cudaFree(activations);
	cudaFree(execution_values);
	cudaFree(costs);
	cudaFree(gradients);
	cudaDeviceSynchronize();

	return cost;
}

data_t NN::calculate_output_costs(
	CostFunctions cost_function,
	size_t t_count,
	data_t* Y_hat,
	data_t* activations, size_t activations_start,
	data_t* costs, size_t costs_start
)
{
	data_t* cost = 0;
	cudaMalloc(&cost, sizeof(data_t));
	cudaDeviceSynchronize();
	cudaMemset(cost, 0, sizeof(data_t));
	cudaDeviceSynchronize();
	switch (cost_function)
	{
	case CostFunctions::MSE:
		MSE_derivative kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, neuron_count, activations_start, *output_activations_start,
			costs, costs_start,
			Y_hat, output_length
			);
		MSE_cost kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, neuron_count, activations_start, *output_activations_start,
			Y_hat, output_length,
			cost
			);
		break;
	case CostFunctions::log_likelyhood:
		log_likelyhood_derivative kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, activations_start,
			neuron_count, *output_activations_start, output_length,
			costs, costs_start,
			Y_hat
			);
		log_likelyhood_cost kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, neuron_count, activations_start, *output_activations_start,
			Y_hat, output_length,
			cost
			);
		break;
	default:
		break;
	}
	cudaDeviceSynchronize();
	multiply_array kernel(1, 1) (
		cost, 1, 1.0 / (output_length * t_count)
		);
	data_t host_cost = 0;
	cudaDeviceSynchronize();
	cudaMemcpy(&host_cost, cost, sizeof(data_t), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(cost);
	return host_cost;
}

void NN::backpropagate(
	size_t t_count, 
	data_t* costs,
	data_t* activations, 
	data_t* execution_values,
	data_t** gradients,
	gradient_hyperparameters hyperparameters
)
{
	apply_regularizations(
		t_count,
		costs, activations, 
		hyperparameters.regularization
	);

	data_t* derivatives = 0;
	if (!*gradients)
	{
		cudaMalloc(gradients, sizeof(data_t) * t_count * gradient_count);
		cudaMemset(*gradients, 0, sizeof(data_t) * t_count * gradient_count);
	}
	if (derivative_count)
	{
		cudaMalloc(&derivatives, sizeof(data_t) * t_count * derivative_count);
		cudaMemset(derivatives, 0, sizeof(data_t) * t_count * derivative_count);
	}
	cudaDeviceSynchronize();

	size_t activations_start = 0;
	size_t execution_values_start = 0;
	size_t derivatives_start = 0;
	size_t gradients_start = 0;
	for (size_t t = 0; t < t_count; t++)
	{
		activations_start = neuron_count * t;
		derivatives_start = derivative_count * t;
		execution_values_start = execution_value_count * t;
		calculate_derivatives(
			activations, activations_start, 
			derivatives, derivatives_start - derivative_count, derivatives_start,
			execution_values, execution_values_start
		);
	}
	for (int t = t_count - 1; t >= 0; t--)
	{
		gradients_start = gradient_count * t;
		size_t next_gradient_start = gradients_start + gradient_count;
		next_gradient_start -= next_gradient_start * (t == t_count - 1);

		derivatives_start = derivative_count * t;
		activations_start = neuron_count * t;

		calculate_gradients(
			activations, activations_start,
			execution_values, execution_values_start,
			costs, activations_start,
			*gradients, gradients_start, next_gradient_start,
			derivatives, derivatives_start, derivatives_start - derivative_count,
			hyperparameters.dropout_rate
		);
	}

	if (!stateful && contains_recurrent_layers)
		delete_memory();
	if (derivative_count) cudaFree(derivatives);
}

void NN::apply_regularizations(
	size_t t_count,
	data_t* costs, data_t *activations,
	regularization_hyperparameters hyperparameters
)
{
	entropy_regularization(
		t_count, neuron_count, output_length, 
		costs, activations, *output_activations_start, 
		hyperparameters.entropy_bonus
	);
}

void NN::calculate_derivatives(
	data_t* activations, size_t activations_start,
	data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
	data_t* execution_values, size_t execution_values_start
)
{
	// Todo: make layer gradient calculation async
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->calculate_derivatives(
			activations, activations_start,
			derivatives, previous_derivatives_start, derivatives_start,
			execution_values, execution_values_start
		);
		cudaDeviceSynchronize();
	}
}

void NN::calculate_gradients(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start,
	data_t* costs, size_t costs_start, 
	data_t* gradients, size_t gradients_start, size_t next_gradients_start, 
	data_t* derivatives, size_t derivatives_start, size_t previous_derivatives_start,
	float dropout_rate
)
{
	for (int i = layer_count - 1; i >= 0; i--)
	{
		size_t layer_len = layers[i]->get_neuron_count();

		float *random_sample = 0;
		cudaMalloc(&random_sample, sizeof(float) * layer_len);
		generate_random_values(random_sample, layer_len, 0, 1);

		short *dropout = 0;
		cudaMalloc(&dropout, sizeof(short) * layer_len);
		cudaMemset(dropout, 0, sizeof(short) * layer_len);

		if (i == layer_count - 1)
			add_to_array kernel(layer_len / 32 + (layer_len % 32 > 0), 32) (dropout, layer_len, 1);
		else
			cud_set_dropout kernel(layer_len / 32 + (layer_len % 32 > 0), 32) (
				dropout_rate, random_sample, dropout, layer_len
			);
		cudaDeviceSynchronize();
		
		element_wise_multiply kernel(layer_len / 32 + (layer_len % 32 > 0), 32) (
			costs + costs_start + layers[i]->layer_activations_start, dropout, layer_len
		);
		cudaDeviceSynchronize();
		cudaFree(random_sample);
		cudaFree(dropout);

		layers[i]->calculate_gradients(
			activations, activations_start,
			execution_values, execution_values_start,
			derivatives, derivatives_start,
			gradients, next_gradients_start, gradients_start,
			costs, costs_start
		);
		cudaDeviceSynchronize();
	}
}

void NN::subtract_gradients(data_t* gradients, size_t gradients_start, gradient_hyperparameters hyperparamters)
{
	reset_NaNs kernel(gradient_count / 32 + (gradient_count % 32 > 0), 32) (
		gradients + gradients_start, 0, gradient_count
	);
	cudaDeviceSynchronize();
	
	global_gradient_clip(gradients + gradients_start, gradient_count, hyperparamters);
	
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* current_layer = layers[i];
		size_t layer_length = current_layer->get_neuron_count();

		current_layer->subtract_gradients(gradients, gradients_start, hyperparamters);
	}
	cudaDeviceSynchronize();
}

/*data_t* NN::PPO_execute(data_t* X, data_t** initial_states, data_t** trajectory_inputs, data_t** trayectory_outputs, size_t n_executions)
{
	if (!initial_states || !trajectory_inputs || !trayectory_outputs || !X) return 0;

	if (!n_executions) *initial_states = get_hidden_state();
	if (n_executions && (!*trajectory_inputs || !*trayectory_outputs)) return 0;

	data_t* device_output = inference_execute(X, cuda_pointer_output);
	if (!device_output) return 0;

	*trajectory_inputs = cuda_append_array(*trajectory_inputs, n_executions * input_length, X, input_length, true, true);
	*trayectory_outputs = cuda_append_array(*trayectory_outputs, n_executions * output_length, device_output, output_length, true);

	data_t* host_output = new data_t[output_length];
	cudaMemcpy(host_output, device_output, sizeof(data_t) * output_length, cudaMemcpyDeviceToHost);
	cudaFree(device_output);
	return host_output;
}

void NN::PPO_train(
	size_t t_count,
	data_t** initial_states, data_t** trajectory_inputs, data_t** trajectory_outputs,
	data_t* rewards, bool are_rewards_at_host, NN* value_function_estimator,
	PPO_hyperparameters hyperparameters
)
{
	if (!initial_states
		|| !trajectory_inputs || !*trajectory_inputs
		|| !trajectory_outputs || !*trajectory_outputs
		|| !rewards || !value_function_estimator)
		return;

	NN* tmp_n = clone();

	data_t* advantages = calculate_advantage(
		t_count,
		value_function_estimator, *trajectory_inputs,
		hyperparameters.GAE, false, false,
		rewards, are_rewards_at_host, false);
	if (!advantages) return;

	int stop = false;
	data_t total_kl_divergence = 0;
	data_t* collected_gradients = 0;
	size_t i = 0;
	for (i = 0; i < hyperparameters.max_training_steps && !stop; i++)
	{
		tmp_n->set_hidden_state(*initial_states, false);

		data_t* execution_values = 0;
		data_t* activations = 0;
		data_t* Y = 0;
		tmp_n->training_execute(t_count, *trajectory_inputs, &Y, cuda_pointer_output, &execution_values, &activations);
		if (!Y) throw;

		data_t* costs = 0;
		cudaMalloc(&costs, sizeof(data_t) * neuron_count * t_count);
		cudaMemset(costs, 0, sizeof(data_t) * neuron_count * t_count);

		total_kl_divergence += PPO_derivative(
			t_count, output_length, neuron_count,
			*trajectory_outputs, Y, advantages,
			costs, *output_activations_start,
			hyperparameters.clip_ratio
		);
		stop = fabs(total_kl_divergence / (i + 1)) > hyperparameters.max_kl_divergence_threshold;
		cudaFree(Y);

		data_t* gradients = 0;
		tmp_n->backpropagate(
			t_count,
			costs, activations, execution_values, &gradients, hyperparameters.policy
		);
		cudaFree(costs);
		cudaFree(activations);
		cudaFree(execution_values);

		for (size_t t = 0; t < t_count; t++)
			tmp_n->subtract_gradients(gradients, gradient_count * t, hyperparameters.policy);
		collected_gradients = cuda_append_array(collected_gradients, gradient_count * t_count * i,
			gradients, gradient_count * t_count, true);
		cudaFree(gradients);
	}

	for (size_t i = 0; i < t_count; i++)
		subtract_gradients(collected_gradients, gradient_count * i, hyperparameters.policy);
	cudaFree(collected_gradients);

	cudaFree(advantages);

	cudaFree(*initial_states);
	*initial_states = 0;

	cudaFree(*trajectory_inputs);
	*trajectory_inputs = 0;

	cudaFree(*trajectory_outputs);
	*trajectory_outputs = 0;

	delete tmp_n;
}*/

data_t* NN::get_hidden_state(size_t *arr_value_count)
{
	size_t current_array_len = 0;
	data_t* out = 0;
	for (size_t i = 0; i < layer_count; i++)
	{
		size_t layer_state_count = layers[i]->get_neuron_count() * layers[i]->hidden_states_per_neuron;
		size_t new_array_len = current_array_len + layer_state_count;
		if (new_array_len == current_array_len) continue;

		out = cuda_realloc(out, current_array_len, new_array_len, true);

		data_t *tmp = layers[i]->get_state();
		if (!tmp)
		{
			cudaFree(out);
			return 0;
		}

		cudaMemcpy(out + current_array_len, tmp, sizeof(data_t) * layer_state_count, cudaMemcpyDeviceToDevice);
		cudaFree(tmp);

		current_array_len = new_array_len;
	}
	if (arr_value_count) *arr_value_count = current_array_len;
	return out;
}

void NN::set_hidden_state(data_t* state, int free_input_state)
{
	if (!state) return;
	size_t state_i = 0;
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* layer = layers[i];

		layer->set_state(state + state_i);

		state_i += layer->hidden_states_per_neuron * layer->get_neuron_count();
	}
	if (free_input_state) cudaFree(state);
}

void NN::evolve()
{
#ifndef DETERMINISTIC
	srand((unsigned int)get_arbitrary_number());
#endif

	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->mutate_fields(evolution_values);
		layers[i]->connections->mutate_fields(evolution_values);
	}
	if (evolution_values.layer_addition_probability > get_random_float())
	{
		printf("Adding layer\n");
		add_layer();
	}
	if (evolution_values.neuron_deletion_probability > get_random_float() && layer_count > 1)
	{
		size_t layer_i = rand() % (layer_count - 1);
		printf("removing neuron at %i\n", (int)layer_i);
		remove_neuron(layer_i);
	}
	if (evolution_values.neuron_addition_probability > get_random_float() && layer_count > 1)
	{
		size_t layer_i = rand() % (layer_count - 1);
		printf("adding_neuron at %i\n", (int)layer_i);
		add_neuron(layer_i);
	}
	float* evolution_values_pointer = (float*)(&evolution_values);
	for (size_t i = 0; i < sizeof(evolution_metadata) / sizeof(float); i++)
	{
		float sign = (1 - 2 * (get_random_float() > .5));
		float mutation = get_random_float() * evolution_values.evolution_metadata_field_max_mutation * sign;
		int will_mutate = evolution_values.evolution_metadata_field_mutation_chance > get_random_float();

		if (will_mutate && evolution_values_pointer[i] + mutation > 1E-5 && evolution_values_pointer[i] + mutation < .3)
			evolution_values_pointer[i] += mutation;
	}
}

void NN::add_layer()
{
	size_t insert_i = layer_count > 1 ? rand() % (layer_count - 1) : 0;
	add_layer(insert_i);
}

void NN::add_layer(size_t insert_i)
{
	NeuronTypes insert_type = (NeuronTypes)(rand() % NeuronTypes::last_neuron_entry);
	add_layer(insert_i, insert_type);
}

void NN::add_layer(size_t insert_i, NeuronTypes layer_type)
{
	weight_init.time = get_arbitrary_number();
	bias_init.time = get_arbitrary_number();
	layer_weights_init.time = get_arbitrary_number();

	size_t previous_layer_length = input_length;
	size_t previous_layer_activations_start = 0;
	if (insert_i)
	{
		ILayer* previous_layer = layers[insert_i];
		previous_layer_length = previous_layer->get_neuron_count();
		previous_layer_activations_start = previous_layer->layer_activations_start;
	}

	weight_init.layer_n_inputs = previous_layer_length;
	weight_init.layer_n_outputs = 1;

	bias_init.layer_n_inputs = previous_layer_length;
	bias_init.layer_n_outputs = 1;
	
	layer_weights_init.layer_n_inputs = previous_layer_length;
	layer_weights_init.layer_n_outputs = 1;


	IConnections* new_connections = new NeatConnections(
		previous_layer_activations_start, previous_layer_length, 1,
		weight_init, bias_init
	);
	ILayer* new_layer = 0;

	switch (layer_type)
	{
	case NeuronTypes::Neuron:
		new_layer = new NeuronLayer(new_connections, 1, (ActivationFunctions)(rand() % ActivationFunctions::activations_last_entry));
		break;
	case NeuronTypes::LSTM:
		new_layer = new LSTMLayer(new_connections, 1, layer_weights_init);
		break;
	default:
		throw "Neuron_type not added to evolve method";
		break;
	}
	new_layer->optimizer = Optimizers(new_layer->get_weight_count(), optimizer_initialization);
	add_layer(insert_i, new_layer);
}

void NN::add_layer(size_t insert_i, ILayer* layer)
{
	ILayer** tmp_layers = layers;
	layer_count++;

	// insert layer
	layers = new ILayer * [layer_count];
	for (size_t i = 0; i < insert_i; i++)
		layers[i] = tmp_layers[i];
	layers[insert_i] = layer;
	for (size_t i = insert_i + 1; i < layer_count; i++)
		layers[i] = tmp_layers[i - 1];

	// Update info
	set_fields();
	size_t added_neuron_count = layer->get_neuron_count();
	size_t added_layer_activations_start = layer->layer_activations_start;
	for (size_t i = 0; i < added_neuron_count; i++)
	{
		adjust_to_added_neuron(insert_i, added_layer_activations_start + i);
	}
	set_fields();
}

void NN::add_output_neuron()
{
	add_neuron(layer_count - 1);
}

void NN::add_input_neuron()
{
	for (size_t i = 0; i < layer_count; i++)
	{
		adjust_to_added_neuron(-1, input_length);
	}
	input_length++;
	set_fields();
}

void NN::add_neuron(size_t layer_i)
{

	size_t previous_layer_length = input_length;
	size_t previous_layer_activations_start = 0;
	if (layer_i)
	{
		ILayer *previous_layer = layers[layer_i];
		previous_layer_length = previous_layer->get_neuron_count();
		previous_layer_activations_start = previous_layer->layer_activations_start;
	}
	size_t added_neuron_i = layers[layer_i]->layer_activations_start + layers[layer_i]->get_neuron_count();

	layers[layer_i]->add_neuron(previous_layer_length, previous_layer_activations_start, 1, 0);
	cudaDeviceSynchronize();

	adjust_to_added_neuron(layer_i, added_neuron_i);
	set_fields();
}

void NN::adjust_to_added_neuron(int layer_i, size_t neuron_i)
{
	size_t layer_distance_from_added_neuron = 1;
	for (int i = layer_i + 1; i < layer_count; i++, layer_distance_from_added_neuron++)
	{
		float connection_probability = 1.0 / layer_distance_from_added_neuron;
		connection_probability += (1 - connection_probability) * evolution_values.layer_distance_from_added_neuron_connection_addition_modifier;

		size_t old_parameter_count = layers[i]->get_weight_count();
		layers[i]->adjust_to_added_neuron(neuron_i, connection_probability);
		size_t new_parameter_count = layers[i]->get_weight_count();
	}
	cudaDeviceSynchronize();
}

void NN::remove_neuron(size_t layer_i)
{
	if (layers[layer_i]->get_neuron_count() == 1)
		return;
	size_t layer_neuron_count = layers[layer_i]->get_neuron_count();
	remove_neuron(layer_i, rand() % layer_neuron_count);
}

void NN::remove_neuron(size_t layer_i, size_t layer_neuron_i)
{
	size_t removed_neuron_i = layers[layer_i]->layer_activations_start + layer_neuron_i;
	layers[layer_i]->remove_neuron(layer_neuron_i);
	for (size_t i = layer_i + 1; i < layer_count; i++)
	{
		size_t old_param_count = layers[i]->get_weight_count();
		layers[i]->adjust_to_removed_neuron(removed_neuron_i);
		size_t new_param_count = layers[i]->get_weight_count();
	}
	cudaDeviceSynchronize();
	set_fields();
}

void NN::delete_memory()
{
	for (size_t i = 0; i < layer_count && contains_recurrent_layers; i++)
		layers[i]->delete_memory();
}

NN* NN::clone()
{
	NN* clone = new NN();
	clone->layer_count = layer_count;
	clone->neuron_count = neuron_count;
	clone->input_length = input_length;
	clone->output_length = output_length;
	
	clone->layers = new ILayer*[layer_count];
	for (size_t i = 0; i < layer_count; i++)
	{
		clone->layers[i] = layers[i]->layer_specific_clone();
		layers[i]->ILayerClone(clone->layers[i]);
	}
	clone->set_fields();
	clone->evolution_values = evolution_values;
	clone->contains_recurrent_layers = contains_recurrent_layers;
	return clone;
}

void NN::save(const char *pathname)
{
	FILE *file = fopen(pathname, "wb");
	if (!file)
	{
		std::cerr << "Could not open file for saving network" << std::endl;
		return;
	}
	save(file);
	fclose(file);
}

void NN::save(FILE* file)
{
	fwrite(&layer_count, sizeof(size_t), 1, file);
	fwrite(&input_length, sizeof(size_t), 1, file);
	for (size_t i = 0; i < layer_count; i++)
	{
		size_t layer_type = (size_t)layers[i]->layer_type;
		fwrite(&layer_type, sizeof(size_t), 1, file);
	}

	for (size_t i = 0; i < layer_count; i++)
	{
		size_t connection_type = (size_t)layers[i]->connections->connection_type;
		fwrite(&connection_type, sizeof(size_t), 1, file);
	}

	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->connections->save(file);
		layers[i]->save(file);
	}
}

NN* NN::load(const char *pathname, bool load_state)
{
	FILE *file = fopen(pathname, "rb");
	if (!file)
		return 0;
	NN *out = load(file);
	fclose(file);

	if (!load_state) out->delete_memory();
	return out;
}

NN* NN::load(FILE* file)
{
	NN* output = new NN();

	fread(&(output->layer_count), sizeof(size_t), 1, file);
	fread(&(output->input_length), sizeof(size_t), 1, file);

	size_t layer_count = output->layer_count;

	NeuronTypes *neuron_types = new NeuronTypes[layer_count];
	ConnectionTypes *connection_types = new ConnectionTypes[layer_count];

	ILayer **output_layers = new ILayer*[layer_count];

	fread(neuron_types, sizeof(NeuronTypes), layer_count, file);
	fread(connection_types, sizeof(ConnectionTypes), layer_count, file);

	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer *layer = 0;
		IConnections *connections = 0;
		switch (neuron_types[i])
		{
			case NeuronTypes::Neuron:
				layer = new NeuronLayer();
				break;
			case NeuronTypes::LSTM:
				layer = new LSTMLayer();
				break;
			default:
				break;
		}
		switch (connection_types[i])
		{
			case ConnectionTypes::Dense:
				connections = new DenseConnections();
				break;
			case ConnectionTypes::NEAT:
				connections = new NeatConnections();
				break;
			default:
				break;
		}
		connections->load(file);
		layer->connections = connections;
		layer->load(file);

		output_layers[i] = layer;
	}

	delete[] connection_types;
	delete[] neuron_types;

	output->layers = output_layers;
	output->set_fields();
	return output;
}

void NN::deallocate()
{
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->deallocate();
		delete layers[i];
	}
	delete[] layers;
}

void NN::print_shape()
{
	printf("%i ", (int)input_length);
	for (size_t i = 0; i < layer_count; i++)
		printf("%i ", (int)layers[i]->get_neuron_count());
	printf("\n");
}


#endif
