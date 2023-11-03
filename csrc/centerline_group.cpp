#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/centerline_group_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__centerline_group_cuda(void) { return NULL; }
#endif
#endif

torch::Tensor centerline_group(torch::Tensor x, torch::Tensor y,
                     torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y,
                     int64_t max_num_neighbors, int64_t num_workers) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return centerline_group_cuda(x, y, ptr_x, ptr_y, max_num_neighbors);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster_topology::centerline_group", &centerline_group);