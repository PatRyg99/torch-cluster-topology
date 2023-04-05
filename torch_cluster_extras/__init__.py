import importlib
import os.path as osp

import torch

__version__ = '1.0.0'

for library in [
        '_version', '_vector_radius', '_centerline_group'
]:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")

cuda_version = torch.ops.torch_cluster_extras.cuda_version()
if torch.cuda.is_available() and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_cluster_extras were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_cluster_extras has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_cluster_extras that '
            f'matches your PyTorch install.')

from .vector_radius import vector_radius # noqa
from .centerline_group import centerline_group  # noqa

__all__ = [
    'vector_radius',
    'centerline_group',
    '__version__',
]
