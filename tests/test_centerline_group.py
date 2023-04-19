from itertools import product

import pytest
import torch
from torch_cluster_extras import centerline_group
from torch_cluster_extras.testing import devices, grad_dtypes, tensor


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_centerline_group(dtype, device):
    x = tensor([
        0, 0, 1, 1, 2, 3, 0, 0, 0, 2, 2, 2
    ], dtype, device).reshape(-1, 1)
    y = tensor([
        [1, 0, 1, 1], 
        [0, 1, 0, 1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0]
    ], dtype, device)

    batch_x = tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 0, 1, 1], torch.long, device)

    edge_index = centerline_group(x, y)
    assert to_set(edge_index) == set([
        (0, 0), (0, 1), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11),
        (1, 2), (1, 3), (1, 5),
        (2, 2), (2, 3), (2, 5),
        (3, 0), (3, 1), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)
    ])

    edge_index = centerline_group(x, y, batch_x, batch_y)
    assert to_set(edge_index) == set([
        (0, 0), (0, 1), (0, 4), (0, 5), 
        (1, 2), (1, 3), (1, 5),
        (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)
    ])

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 0, 2, 2], torch.long, device)
    edge_index = centerline_group(x, y, batch_x, batch_y)
    assert to_set(edge_index) == set([
        (0, 0), (0, 1), (0, 4), (0, 5), 
        (1, 2), (1, 3), (1, 5),
        (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)
    ])
