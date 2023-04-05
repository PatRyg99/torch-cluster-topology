# PyTorch Cluster Extras

--------------------------------------------------------------------------------

This package extends `pytorch-cluster` library with custom graph clustering algorithms fo the use in [PyTorch](http://pytorch.org/).
The package consists of the following clustering algorithms:

* **[Vector Radius](#evg)** clustering based on vector, utilized in Rygiel *et al.*: [Eigenvector Grouping for Point Cloud Vessel Labeling](https://proceedings.mlr.press/v194/rygiel22a/rygiel22a.pdf) (GeoMedIA 2022)
* **[Centerline Grouping](#cg)** from Rygiel *et al.*: CenterlinePointNet++: A new point cloud based architecture for coronary artery pressure drop and vFFR estimation (under review)

All included operations work on varying data types and are implemented for GPU.

## Installation

Clone repository locally, navigate to the root directory and run command below to install package in current python environment:  
`pip install -e .`

## Functions

### Vector Radius

A point clustering algorithm around the line segment (vector) specified by the pair of $N$-dimensional points. 

```python
import torch
from torch_cluster_extras import vector_radius

x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]).cuda()
y = torch.Tensor([[-1, 0, 0, 0], [1, 0, 0, 0]]).cuda()

row, col = vector_radius(x, y, 1.2)
```

```
print(row, col)
tensor([0, 0, 1, 1]) tensor([0, 1, 2, 3])
```

### Centerline Grouping

A point clustering algorithm that groups surface points based on the underlying centerline.
The surface points are provided as the mapping ids to the centerline, and centerline is provided
as the thresholded all pair graph distance matrix (`1` are considered to be neighbours for the grouping and `O` not).

```python
import torch
from torch_cluster_extras import centerline_group

x = torch.Tensor([[0], [0], [1], [1], [2], [3], [4]]).cuda()
y = torch.Tensor([[1, 0, 1, 1, 0]]).cuda()

row, col = centerline_group(x, y)
```

```
print(row, col)
tensor([0, 0, 0, 0]) tensor([0, 1, 4, 5])
```
