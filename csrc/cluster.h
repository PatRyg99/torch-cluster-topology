#pragma once

#include <torch/extension.h>

int64_t cuda_version();

torch::Tensor vector_radius(torch::Tensor x, torch::Tensor y,
                     torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y, double r,
                     int64_t max_num_neighbors, int64_t num_workers)

torch::Tensor centerline_group(torch::Tensor x, torch::Tensor y,
                     torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y,
                     int64_t max_num_neighbors, int64_t num_workers)