#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/vector_radius_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__vector_radius_cuda(void) { return NULL; }
#endif
#endif

torch::Tensor vector_radius(torch::Tensor x, torch::Tensor y,
                     torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y, double r,
                     int64_t max_num_neighbors, int64_t num_workers) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return vector_radius_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster_extras::vector_radius", &vector_radius);