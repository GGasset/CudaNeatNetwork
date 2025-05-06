#pragma once

typedef struct evolution_metadata {
	float field_max_evolution = .002;
	float field_mutation_chance = .01;
	float layer_addition_probability = .01;
	float neuron_deletion_probability = .01;
	float neuron_addition_probability = .02;
	float layer_distance_from_added_neuron_connection_addition_modifier = .001;

	float evolution_metadata_field_mutation_chance = .005;
	float evolution_metadata_field_max_mutation = .001;
} evolution_metadata;
