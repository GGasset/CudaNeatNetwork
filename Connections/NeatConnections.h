#pragma once
#include "IConnections.h"
#include "vector"

class NeatConnections :
    public IConnections
{
public:
    /// <summary>
    /// Device array
    /// </summary>
    size_t* connection_points = 0;
    /// <summary>
    /// Host array
    /// </summary>
    //size_t* connection_counts = 0;
    
    /// <summary>
    /// Device array
    /// </summary>
    size_t* connection_neuron_i = 0;

    NeatConnections(size_t previous_layer_start, size_t previous_layer_length, size_t neuron_count);
    NeatConnections();

    void linear_function(size_t activations_start, data_t* activations,
        data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
    ) override;

    void calculate_derivative(
        size_t activations_start, data_t* activations,
        size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
    ) override;

    void calculate_gradients(
        data_t* activations, size_t activations_start,
        data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
        data_t* costs, size_t costs_start
    ) override;

    void subtract_gradients(
        data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
        data_t learning_rate, data_t gradient_clip
    ) override;

    size_t get_connection_count_at(size_t neuron_i) override;
	
    IConnections* connections_specific_clone() override;

    void add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections) override;
    void adjust_to_added_neuron(size_t added_neuron_i, float connection_probability, std::vector<size_t>* added_connections_neuron_i) override;
    void remove_neuron(size_t neuron_i) override;
    void adjust_to_removed_neuron(size_t neuron_i, std::vector<size_t>* removed_connections_neuron_i) override;

    void specific_save(FILE* file) override;
    void load(FILE* file) override;

    void specific_deallocate() override;
};

