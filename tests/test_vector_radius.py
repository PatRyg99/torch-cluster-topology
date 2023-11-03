from itertools import product

import pytest
import torch
from torch_cluster_topology import vector_radius
from torch_cluster_topology.testing import devices, grad_dtypes, tensor


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_vector_radius(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    y = tensor([
        [-0.5, -0.5, -0.5, +0.5], 
        [-0.5, +0.5, +0.5, +0.5], 
        [+0.5, +0.5, +0.5, -0.5], 
        [+0.5, -0.5, -0.5, -0.5], 
    ], dtype, device)

    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 0, 1, 1], torch.long, device)

    edge_index = vector_radius(x, y, 1.0)
    assert to_set(edge_index) == set([
        (0, 0), (0, 4), (0, 1), (0, 5),
        (1, 1), (1, 5), (1, 2), (1, 6),
        (2, 2), (2, 6), (2, 3), (2, 7),
        (3, 3), (3, 7), (3, 0), (3, 4)
    ])

    edge_index = vector_radius(x, y, 1.0, batch_x, batch_y)
    assert to_set(edge_index) == set([
        (0, 0), (0, 1),
        (1, 1), (1, 2),
        (2, 6), (2, 7),
        (3, 7), (3, 4)
    ])

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 0, 2, 2], torch.long, device)
    edge_index = vector_radius(x, y, 1.0, batch_x, batch_y)
    assert to_set(edge_index) == set([
        (0, 0), (0, 1),
        (1, 1), (1, 2),
        (2, 6), (2, 7),
        (3, 7), (3, 4)
    ])
