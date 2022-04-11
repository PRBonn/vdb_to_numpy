from typing import Tuple

import numpy as np
import pyopenvdb as vdb

from ..pybind import vdb_pybind


class LeafNodeGrid:
    """LeafNodeGrid is basically a wrapper around pyopenvdb.FloatGrid but instead of operating on
    the entire grid it only acess the leaf nodes of the original grid.

    Some functionallity will be lost in the way, but this class is only intended to use
    for getting training data
    """

    def __init__(self, vdb_grid: vdb.FloatGrid, normalize: bool = False):
        """Convert a pyopenvdb.FloatGrid to a Numpy-based LeafNodeGrid."""
        if not isinstance(vdb_grid, vdb.FloatGrid):
            raise ValueError("GridType: '{}' not supported".format(type(vdb_grid)))
        self.vdb_grid = vdb_grid.copy()
        self.leaf_nodes = vdb_pybind.extract_leaf_nodes(self.vdb_grid)
        self.voxel_size = np.float32(self.vdb_grid.transform.voxelSize()[0])
        self.background = np.float32(self.vdb_grid.background)
        self.transform = self.vdb_grid.transform
        self.gridClass = self.vdb_grid.gridClass

        # Normalize the leaf_nodes_a array if specified
        if normalize:
            for _, leaf_node in self.leaf_nodes:
                leaf_node /= self.background

        # Cache the array format for the LeafNodeGrid
        if len(self.leaf_nodes) != 0:
            coords_ijk, nodes = list(zip(*self.leaf_nodes))
            self.coords_ijk_a = np.asarray(coords_ijk, dtype=np.int32)
            self.leaf_nodes_a = np.asarray(nodes, dtype=np.float32)

    def numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the current represetantion of the LeafNode grid to stacked numpy arrays."""
        return self.coords_ijk_a.copy(), self.leaf_nodes_a.copy()

    def to_vdb(self):
        """Convert to vdb format."""
        vdb_grid = vdb.FloatGrid()
        vdb_grid.transform = self.transform
        vdb_grid.background = self.background
        vdb_grid.gridClass = self.gridClass
        coords_ijk, leaf_nodes = self.numpy()
        for i, ijk in enumerate(coords_ijk):
            vdb_grid.copyFromArray(leaf_nodes[i], ijk)
        return vdb_grid

    def __len__(self):
        return len(self.leaf_nodes)

    def __getitem__(self, idx):
        """Returns a tuple of (coord_ijk, leaf_node_buffer)."""
        if idx > len(self):
            raise IndexError("idx:{} >= max_idx:{}".format(idx, len(self)))
        return self.leaf_nodes[idx]

    def __iter__(self):
        return self.leaf_nodes.__iter__()

    def __next__(self):
        return self.leaf_nodes.__next__()

    @property
    def leaf_node_shape(self):
        return self.leaf_nodes[0][1].shape

    @property
    def sdf_trunc(self):
        return np.float32(self.background)
