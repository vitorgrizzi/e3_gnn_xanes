"""
Graph transforms for XANES data preprocessing.
"""
import torch_geometric.transforms as T


class RadiusGraphTransform:
    """
    Wraps PyG's RadiusGraph to build edge_index from atomic positions
    using a distance cutoff. Use as a pre_transform when building datasets.
    
    Example:
        transform = RadiusGraphTransform(r_max=5.0)
        data = transform(data)  # adds edge_index based on radius cutoff
    """
    def __init__(self, r_max=5.0, loop=False, max_num_neighbors=32):
        self.transform = T.RadiusGraph(
            r=r_max, loop=loop, max_num_neighbors=max_num_neighbors
        )

    def __call__(self, data):
        return self.transform(data)

    def __repr__(self):
        return f"{self.__class__.__name__}(r={self.transform.r})"
