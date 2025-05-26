#ifndef NN_DEFINITIONS
#define NN_DEFINITIONS

#include "NN.h"

size_t NN::get_input_length()
{
	return input_length; 
}

size_t NN::get_output_length()
{
	return output_length;
}

NN::NN(ILayer** layers, size_t input_length, size_t layer_count)
{
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

void NN::execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, short copy_output_to_host = true)
{
	cudaMemcpy(activations + t * neuron_count, input + input_length * t, sizeof(data_t) * input_length, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->execute(activations, neuron_count * t, execution_values, execution_value_count * t);
		cudaDeviceSynchronize();
	}
	if (copy_output_to_host)
	{
		cudaMemcpy(output_start_pointer + output_length * t, activations + neuron_count * t + *output_activations_start, sizeof(data_t) * output_length, cudaMemcpyDeviceToHost);
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

data_t* NN::batch_execute(data_t* input, size_t t_count)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	set_up_execution_arrays(&execution_values, &activations, t_count);

	data_t* outputs = new data_t[output_length * t_count];
	for (size_t i = 0; i < output_length * t_count; i++)
	{
		outputs[i] = 0;
	}
	for (size_t t = 0; t < t_count; t++)
	{
		execute(input, execution_values, activations, t, outputs, 1);
	}


	cudaFree(execution_values);
	cudaFree(activations);
	cudaDeviceSynchronize();
	return outputs;
}

data_t* NN::inference_execute(data_t* input)
{
	return batch_execute(input, 1);
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
	case CostFunctions::PPO:
		PPO_cost kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, activations_start, 
			neuron_count, *output_activations_start, output_length,
			costs, costs_start,
			Y_hat
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

void NN::training_execute(
	size_t t_count,
	data_t* X,
	data_t** Y,
	bool copy_Y_to_host,
	data_t** execution_values,
	data_t** activations,
	size_t arrays_t_length
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


	if (copy_Y_to_host)
	{
		*Y = new data_t[output_length * t_count];
		for (size_t i = 0; i < output_length * t_count; i++)
		{
			(*Y)[i] = 0;
		}
	}

	for (size_t t = 0; t < t_count; t++)
	{
		execute(X, (*execution_values) + execution_value_count * arrays_t_length, (*activations) + neuron_count * arrays_t_length, t, copy_Y_to_host ? *Y : 0, copy_Y_to_host);
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
		t_count, costs, activations, execution_values, &gradients, hyperparameters.dropout_rate
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

data_t NN::training_batch(
	size_t t_count,
	data_t* X,
	data_t* Y_hat,
	bool is_Y_hat_on_host_memory,
	size_t Y_hat_value_count,
	CostFunctions cost_function,
	data_t** Y,
	bool copy_Y_to_host,
	gradient_hyperparameters hyperparameters
)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	training_execute(
		t_count,
		X,
		Y,
		copy_Y_to_host,
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

void NN::backpropagate(
	size_t t_count, 
	data_t* costs,
	data_t* activations, 
	data_t* execution_values,
	data_t** gradients,
	float	dropout_rate
)
{
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
			dropout_rate
		);
	}

	if (!stateful && contains_recurrent_layers)
		delete_memory();
	if (derivative_count) cudaFree(derivatives);
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
		IConnections::generate_random_values(&random_sample, layer_len, 0, 1);

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

/*data_t* NN::calculate_GAE_advantage(size_t t_count, data_t gamma, data_t lambda, NN* value_function_estimator, data_t* value_function_state, data_t* rewards)
{
	return nullptr;
}*/

data_t *NN::calculate_GAE_advantage(
	size_t t_count,
	data_t gamma, data_t lambda,
	NN *value_function_estimator, data_t *value_function_state, gradient_hyperparameters estimator_hyperparameters, bool is_state_on_host, bool free_state,
	data_t *rewards, bool is_reward_on_host, bool free_rewards
)
{
	if (!value_function_estimator) return (0);

	data_t *discounted_rewards = 0;
	cudaMalloc(&discounted_rewards, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();
	if (!discounted_rewards) return (0);

	cudaMemset(discounted_rewards, 0, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();

	calculate_discounted_rewards kernel(t_count / 32 + (t_count % 32 > 0), 32) (
		t_count, gamma, rewards, discounted_rewards
	);
	cudaDeviceSynchronize();

	data_t *host_value_functions = 0;
	value_function_estimator->training_batch( // TODO Returns a pointer to host memory and then copies it. Slow!
		t_count,
		value_function_state, discounted_rewards, 0, t_count,
		CostFunctions::MSE, 
		&host_value_functions, 1, estimator_hyperparameters
	);

	data_t* value_functions = 0;
	cudaMalloc(&value_functions, sizeof(data_t) * value_function_estimator->get_output_length() * t_count);
	cudaDeviceSynchronize();
	if (!value_functions || !host_value_functions)
		return 0;
	cudaMemcpy(value_functions, host_value_functions, sizeof(data_t) * value_function_estimator->get_output_length() * t_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	delete[] host_value_functions;

	data_t *deltas = 0;
	cudaMalloc(&deltas, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();
	if (!deltas) return (0);

	cudaMemset(deltas, 0, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();

	calculate_deltas kernel(t_count / 32 + (t_count % 32 > 0), 32) (
		t_count, 
		gamma, 
		rewards, 
		value_functions, 
		deltas
	);
	cudaDeviceSynchronize();

	data_t* advantages = 0;
	cudaMalloc(&advantages, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();
	if (!advantages) return 0;

	cudaMemset(advantages, 0, sizeof(data_t) * t_count);
	cudaDeviceSynchronize();

	parallel_calculate_GAE_advantage kernel(t_count / 32 + (t_count % 32 > 0), 32) (
		t_count,
		gamma, lambda,
		deltas, advantages
	);
	cudaDeviceSynchronize();

	cudaFree(deltas);
	cudaFree(discounted_rewards);
	cudaFree(value_functions);
	delete[] value_functions;

	return 0;
}

void NN::subtract_gradients(data_t* gradients, size_t gradients_start, gradient_hyperparameters hyperparamters)
{
	reset_NaNs kernel(gradient_count / 32 + (gradient_count % 32 > 0), 32) (
		gradients + gradients_start, 0, gradient_count
	);
	cudaDeviceSynchronize();
	
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* current_layer = layers[i];
		size_t layer_length = current_layer->get_neuron_count();

		current_layer->subtract_gradients(gradients, gradients_start, hyperparamters);
	}
	cudaDeviceSynchronize();
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
		NeuronTypes insert_type = (NeuronTypes)(rand() % NeuronTypes::last_neuron_entry);
		size_t insert_i = layer_count > 1 ? rand() % (layer_count - 1) : 0;
		
		size_t previous_layer_length = input_length;
		size_t previous_layer_activations_start = 0;
		if (insert_i)
		{
			ILayer* previous_layer = layers[insert_i];
			previous_layer_length = previous_layer->get_neuron_count();
			previous_layer_activations_start = previous_layer->layer_activations_start;
		}
		
		IConnections* new_connections = new NeatConnections(previous_layer_activations_start, previous_layer_length, 1);
		ILayer* new_layer = 0;

		switch (insert_type)
		{
		case NeuronTypes::Neuron:
			new_layer = new NeuronLayer(new_connections, 1, (ActivationFunctions)(rand() % ActivationFunctions::activations_last_entry));
			break;
		case NeuronTypes::LSTM:
			new_layer = new LSTMLayer(new_connections, 1);
			break;
		default:
			throw "Neuron_type not added to evolve method";
			break;
		}
		new_layer->optimizer = host_optimizer_init(default_optimizer, new_layer->get_weight_count());
		add_layer(insert_i, new_layer);
	}
	if (evolution_values.neuron_deletion_probability > get_random_float() && layer_count > 1)
	{
		printf("removing neuron\n");
		size_t layer_i = rand() % (layer_count - 1);
		remove_neuron(layer_i);
	}
	if (evolution_values.neuron_addition_probability > get_random_float() && layer_count > 1)
	{
		printf("adding_neuron\n");
		size_t layer_i = rand() % (layer_count - 1);
		add_neuron(layer_i);
	}
	float* evolution_values_pointer = (float*)(&evolution_values);
	for (size_t i = 0; i < sizeof(evolution_metadata) / sizeof(float); i++)
	{
		evolution_values_pointer[i] +=
			evolution_values.evolution_metadata_field_max_mutation *
			(evolution_values.evolution_metadata_field_mutation_chance > get_random_float()) *
			(1 - 2 * (get_random_float() > .5));
	}
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
	call_optimizer_values_alloc kernel(1, 1) (layers[layer_i]->optimizer, layers[layer_i]->get_weight_count());
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
		if (old_parameter_count != new_parameter_count)
			call_optimizer_values_alloc kernel(1, 1) (layers[i]->optimizer, new_parameter_count);
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
	call_optimizer_values_alloc kernel(1, 1) (layers[layer_i]->optimizer, layers[layer_i]->get_weight_count());
	for (size_t i = layer_i + 1; i < layer_count; i++)
	{
		size_t old_param_count = layers[i]->get_weight_count();
		layers[i]->adjust_to_removed_neuron(removed_neuron_i);
		size_t new_param_count = layers[i]->get_weight_count();
		if (old_param_count != new_param_count)
			call_optimizer_values_alloc kernel(1, 1) (layers[i]->optimizer, new_param_count);
	}
	cudaDeviceSynchronize();
	set_fields();
}

void NN::delete_memory()
{
	for (size_t i = 0; i < layer_count; i++)
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
		return;
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
