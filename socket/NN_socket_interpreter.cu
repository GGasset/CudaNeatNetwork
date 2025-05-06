#include "NN_socket_interpreter.h"

NN_manager::NN_manager(size_t bucket_count)
{
	networks = new HashTable<size_t, network_container*>(bucket_count);
}

size_t NN_manager::add_NN(NN *n)
{
	auto ids = networks->GetKeys();
	size_t network_id = 0;
	if (ids)
	{
		network_id = ids->get_max() + 1;
		ids->free();
	}

	network_container* network = (network_container*)malloc(sizeof(network_container));
	network->batch_count = 0;
	network->accumulated_training_t_count = 0;
	network->accumulated_activations = 0;
	network->accumulated_execution_values = 0;
	network->accumulated_Y_hat = 0;

	network->network = n;

	networks->Add(network_id, network);
	network_count++;
	return network_id;
}

return_specifier* NN_manager::parse_message(char* message, size_t message_length)
{
	if (message_length < sizeof(size_t)) throw;
	return_specifier* output = (return_specifier*)malloc(sizeof(return_specifier));
	output->return_value = 0;
	output->value_count = 0;
	output->error = error_values::OK;

	action_enum action = (action_enum)(*(size_t*)message);
	size_t offset = sizeof(size_t);
	switch (action)
	{
		case construct:
			{
				NN_constructor constructor = NN_constructor();
				size_t layer_count = *(size_t*)(message + offset);
				offset += sizeof(size_t);

				if (!layer_count || message_length != sizeof(size_t) * 3 + sizeof(bool) + layer_count * sizeof(size_t) * 4) throw;

				for (size_t i = 0; i < layer_count; i++)
				{
					ConnectionTypes connections = *(ConnectionTypes*)(message + offset);
					offset += sizeof(size_t);

					NeuronTypes neurons = *(NeuronTypes*)(message + offset);
					offset += sizeof(size_t);

					size_t neuron_count = *(size_t*)(message + offset);
					offset += sizeof(size_t);

					ActivationFunctions activation = *(ActivationFunctions*)(message + offset);
					offset += sizeof(size_t);

					constructor.append_layer(connections, neurons, neuron_count, activation);
				}
				size_t input_length = *(size_t*)(message + offset);
				offset += sizeof(size_t);

				bool stateful = *(bool*)(message + offset);
				offset += sizeof(bool);
				
				size_t network_id = add_NN(constructor.construct(input_length, stateful));

				output->return_value = new data_t[1];
				output->return_value[0] = network_id;
				output->value_count = 1;
#ifdef log_positive
				char log[] = "Network created\n";
				std::cout << log << std::endl;
#endif
			}
			break;
		case destruct:
			{
				if (message_length != sizeof(size_t) * 2) throw;

				size_t id = *(size_t*)(message + offset);
				offset += sizeof(size_t);

				bool is_registered = false;
				network_container* network = networks->Get(id, is_registered);
				if (is_registered)
				{
					if (network->accumulated_activations) cudaFree(network->accumulated_activations);
					if (network->accumulated_execution_values) cudaFree(network->accumulated_execution_values);
					if (network->accumulated_Y_hat) cudaFree(network->accumulated_Y_hat);
					delete network->network;
					free(network);
#ifdef log_positive
					char log[] = "Network destructed";
					std::cout << log << std::endl;
#endif
				}
				else
				{
					char error[] = "Error: id not found while destructing";
					output->error = error_values::NN_not_found;
				}
				networks->Remove(id);
			}
			break;
		case save:
			{
				size_t id = *(size_t *)(message + offset);
				offset += sizeof(size_t);

				size_t path_name_length = *(size_t *)(message + offset);
				offset += sizeof(size_t);
				if (message_length != path_name_length + sizeof(size_t) * 3) throw;

				char *path_name = new char[path_name_length + 1];
				path_name[path_name_length] = 0;
				for (size_t i = 0; i <= path_name_length; i++)
					path_name[i] = *(char *)(message + offset + i);
				offset += path_name_length;

				bool is_registered = false;
				network_container* network = networks->Get(id, is_registered);
				if (!is_registered)
				{
					char error[] = "Error: Network not found while saving";
					std::cerr << error << std::endl;
					output->error = error_values::NN_not_found;
					break;
				}
				NN *n = network->network;
#ifdef log_positive
				char log[] = "Tried saving network";
				std::cout << log << std::endl;
#endif
				n->save(path_name);
				delete[] path_name;
			}
			break;
		case load:
			{
				size_t path_length = *(size_t *)(message + offset);
				offset += sizeof(size_t);

				bool load_state = *(bool *)(message + offset);
				offset++;


				if (message_length != sizeof(size_t) * 2 + sizeof(bool) + path_length) throw;

				char *path = new char[message_length + 1];
				path[message_length] = 0;
				for (size_t i = 0; i <= path_length; i++)
					path[i] = *(char *)(message + offset + i);
				offset += path_length;

				NN *n = NN::load(path, load_state);
				if (!n)
				{
					char error[] = "Error: network not found while loading";
					std::cerr << error << std::endl;
					output->error = error_values::NN_not_found;
					break;
				}
				output->return_value = new data_t[1];
				output->value_count = 1;
				output->return_value[0] = add_NN(n);
#ifdef log_positive
				char log[] = "Loaded network";
				std::cout << log << std::endl;
#endif
			}
			break;
		default:
			char error[] = "Error: Action not found";
			std::cerr << error << std::endl;
			throw;
	}
	if (offset > message_length) throw;
	return output;
}
