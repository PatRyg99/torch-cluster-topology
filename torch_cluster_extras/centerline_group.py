from typing import Optional

import torch


@torch.jit.script
def centerline_group(x: torch.Tensor, y: torch.Tensor,
           batch_x: Optional[torch.Tensor] = None,
           batch_y: Optional[torch.Tensor] = None, max_num_neighbors: int = 32,
           num_workers: int = 1) -> torch.Tensor:
    r"""Groups surface points given as centerline mappings :obj:`x` based on centerline neighbours matrix :obj:`y`.

    Args:
        x (Tensor): Mappings between surface points and centerline nodes
            :math:`\mathbf{X} \in \mathbb{R}`.
        y (Tensor): Centerline neighbours matrix, where :math:`L` is a centerline length
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times L}`.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)

    .. code-block:: python
    
        import torch
        from torch_cluster_topology import centerline_group

        x = torch.Tensor([[0], [0], [1], [1], [2], [3], [4]]).cuda()
        y = torch.Tensor([[1, 0, 1, 1, 0]]).cuda()

        row, col = centerline_group(x, y)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    batch_size = 1
    if batch_x is not None:
        assert x.size(0) == batch_x.numel()
        batch_size = int(batch_x.max()) + 1
    if batch_y is not None:
        assert y.size(0) == batch_y.numel()
        batch_size = max(batch_size, int(batch_y.max()) + 1)

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None
    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return torch.ops.torch_cluster_topology.centerline_group(x, y, ptr_x, ptr_y,
                                          max_num_neighbors, num_workers)