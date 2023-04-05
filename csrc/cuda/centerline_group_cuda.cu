#include "centerline_group_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256

template <typename scalar_t>
__global__ void
centerline_group_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
              const int64_t *__restrict__ ptr_x,
              const int64_t *__restrict__ ptr_y, 
              int64_t *__restrict__ row, int64_t *__restrict__ col, 
              const int64_t n, const int64_t m, const int64_t offset,
              const int64_t num_examples, const int64_t max_num_neighbors) {

  const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  int64_t count = 0;
  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {

    int64_t centerline_idx = x[n_x];
    int64_t is_neighbour = y[n_y * offset + centerline_idx];
    
    if (is_neighbour == 1) {
      row[n_y * max_num_neighbors + count] = n_y;
      col[n_y * max_num_neighbors + count] = n_x;
      count++;
    }

    if (count >= max_num_neighbors)
      break;
  }
}

torch::Tensor centerline_group_cuda(const torch::Tensor x, const torch::Tensor y,
                          torch::optional<torch::Tensor> ptr_x,
                          torch::optional<torch::Tensor> ptr_y,
                          const int64_t max_num_neighbors) {
  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_INPUT(x.dim() == 2);

  CHECK_CUDA(y);
  CHECK_CONTIGUOUS(y);
  CHECK_INPUT(y.dim() == 2);

  cudaSetDevice(x.get_device());

  if (ptr_x.has_value()) {
    CHECK_CUDA(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  } else
    ptr_x = torch::arange(0, x.size(0) + 1, x.size(0),
                          x.options().dtype(torch::kLong));

  if (ptr_y.has_value()) {
    CHECK_CUDA(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  } else
    ptr_y = torch::arange(0, y.size(0) + 1, y.size(0),
                          y.options().dtype(torch::kLong));

  cudaSetDevice(x.get_device());

  auto row =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());
  auto col =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());

  dim3 BLOCKS((y.size(0) + THREADS - 1) / THREADS);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = x.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
    centerline_group_kernel<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x.value().data_ptr<int64_t>(), ptr_y.value().data_ptr<int64_t>(),
        row.data_ptr<int64_t>(), col.data_ptr<int64_t>(), x.size(0),
        y.size(0), y.size(1),
        ptr_x.value().numel() - 1, max_num_neighbors);
  });

  auto mask = row != -1;
  return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
