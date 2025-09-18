
#include "Optimizers.h"

void Optimizers::copy_values_of(size_t optimizer_i, Optimizers dst)
{
    Optimizer_values dst_values = dst.optimizer_values[optimizer_i];
    Optimizer_values src_values = optimizer_values[optimizer_i];

    if (dst_values.value_count_per_parameter != src_values.value_count_per_parameter) throw;
    if (dst.parameter_count != parameter_count) throw;
    if (!dst_values.values || !src_values.values) return ;

    size_t arr_len = src_values.value_count_per_parameter * parameter_count;
    cudaMemcpy(dst_values.values, src_values.values, sizeof(data_t) * arr_len, cudaMemcpyDeviceToDevice);
}

void Optimizers::save_values_of(size_t optimizer_i, FILE *file)
{
    Optimizer_values optimizer = optimizer_values[optimizer_i];
    size_t len = parameter_count * optimizer.value_count_per_parameter;
    if (!len) return;
    save_array(optimizer.values, len, file, true);
}

Optimizer_values Optimizers::load_values_of(size_t optimizer_i, FILE *file)
{
    Optimizer_values out;

    size_t values_per_param = optimizer_values_per_parameter[optimizer_i];
    out.value_count_per_parameter = values_per_param;
    size_t arr_len = values_per_param * parameter_count;
    if (!arr_len) return out;

    out.values = load_array<data_t>(arr_len, file, true);
}

void Optimizers::initialize_values(long parameter_start_i, size_t parameter_count)
{
    data_t *adam = optimizer_values[Adam].values;
    adam[0] = initialization_hyperparameters.adam.beta_1;
    adam[1] = initialization_hyperparameters.adam.beta_2;
    adam[2] = initialization_hyperparameters.adam.epsilon;
    adam[5] = adam[0];
    adam[6] = adam[1];
}

void Optimizers::allocate_values()
{
    for (size_t i = 0; i < optimizers_enum::last_optimizer_entry; i++)
    {
        if (!optimizer_values[i].value_count_per_parameter)
            continue;
        cudaFree(optimizer_values[i].values);

        size_t len = parameter_count * optimizer_values[i].value_count_per_parameter;
        cudaMalloc(&optimizer_values[i].values, sizeof(data_t) * len);
        cudaMemset(optimizer_values[i].values, 0, sizeof(data_t) * len);
    }
}

Optimizers::Optimizers()
{
    optimizer_values_per_parameter[Adam] = 7;
    optimizer_values_per_parameter[ElasticNet] = 0;

    for (size_t i = 0; i < last_optimizer_entry; i++)
        optimizer_values[i].value_count_per_parameter 
            = optimizer_values_per_parameter[i];
}

Optimizers::Optimizers(size_t parameter_count, optimizers_hyperparameters optimizer_options)
{
    *this = Optimizers();

    this->parameter_count = parameter_count;
    initialization_hyperparameters = optimizer_options;

    allocate_values();
    initialize_values(0, parameter_count);
}

Optimizers Optimizers::Clone()
{
    Optimizers out = Optimizers(parameter_count, initialization_hyperparameters);
    for (size_t i = 0; i < last_optimizer_entry; i++)
        copy_values_of(i, out);
    return out;
}

Optimizers::~Optimizers()
{
    for (size_t i = 0; i < optimizers_enum::last_optimizer_entry; i++)
    {
        cudaFree(optimizer_values[i].values);
        optimizer_values[i].values = 0;
    }
    
}

void Optimizers::set_initialization(optimizers_hyperparameters new_initialization)
{
    initialization_hyperparameters = new_initialization;
}