"""Test the C++/Python bindings."""
from typing import Tuple
import unittest

import numpy as np
import pyopenvdb as vdb

import test_data
from vdb_to_numpy.grid_wrappers import LeafNodeGrid


class LeafNodeGridTest(unittest.TestCase):
    """Test cases for a openvdb::LeafNodeGrid."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This is how each leaf node should look, each leaf is a small voxel
        # grid of 8^3 elments, this is hardcoded in the openvdb::LeafNodeGrid
        # representation.
        self.ref_ijk = np.zeros((3,), dtype=np.int32)
        self.ref_node = np.zeros((8, 8, 8), dtype=np.float32)

    def _test_leaf_nodes(self, grid=None):
        # Extract all leaf nodes
        float_grid = LeafNodeGrid(grid)
        self.assertEqual(len(float_grid), grid.leafCount())

        for coord_ijk, leaf_node in float_grid:
            self.assertEqual(coord_ijk.shape, self.ref_ijk.shape)
            self.assertEqual(leaf_node.shape, self.ref_node.shape)
            self.assertEqual(coord_ijk.strides, self.ref_ijk.strides)
            self.assertEqual(leaf_node.strides, self.ref_node.strides)

    def _assert_equal_grids(self, grid1, grid2):
        """Compare all the VDB grid properties and make sure that both grids
        are the same even when they represent different objects."""
        self.assertEqual(grid1.background, grid2.background)
        self.assertEqual(grid1.gridClass, grid2.gridClass)
        self.assertEqual(grid1.transform, grid2.transform)
        self.assertEqual(grid1.leafCount(), grid2.leafCount())
        self.assertEqual(grid1.nonLeafCount(), grid2.nonLeafCount())
        self.assertEqual(grid1.activeVoxelCount(), grid2.activeVoxelCount())
        self.assertEqual(grid1.evalLeafDim(), grid2.evalLeafDim())
        self.assertEqual(grid1.evalActiveVoxelDim(), grid2.evalActiveVoxelDim())
        self.assertEqual(grid1.activeLeafVoxelCount(), grid2.activeLeafVoxelCount())
        self.assertEqual(grid1.evalLeafBoundingBox(), grid2.evalLeafBoundingBox())
        self.assertEqual(grid1.evalActiveVoxelBoundingBox(), grid2.evalActiveVoxelBoundingBox())
        self.assertEqual(grid1.evalMinMax()[0], grid2.evalMinMax()[0])
        self.assertEqual(grid1.evalMinMax()[1], grid2.evalMinMax()[1])

    def _test_to_vdb(self, grid=None):
        """Create a simple grid, convert it to np, and then back to vdb."""
        self._assert_equal_grids(grid1=grid, grid2=LeafNodeGrid(grid).to_vdb())

    def _test_voxel_size(self, grid=None):
        # Check that the voxel size member is propperly initalized
        float_grid = LeafNodeGrid(grid)
        self.assertEqual(float_grid.voxel_size, grid.transform.voxelSize()[0])

    def _test_get_item(self, grid=None):
        float_grid = LeafNodeGrid(grid)
        self.assertIsNotNone(float_grid[0])
        self.assertIsInstance(float_grid[0], Tuple)
        with self.assertRaises(IndexError):
            float_grid[len(float_grid) + 1]

    def _test_leaf_node_shape(self, grid=None):
        float_grid = LeafNodeGrid(grid)
        self.assertEqual(float_grid.leaf_node_shape, self.ref_node.shape)

    def _test_background_value(self, grid=None):
        float_grid = LeafNodeGrid(grid)
        self.assertEqual(float_grid.sdf_trunc, grid.background)

    def test_bunny(self):
        bunny_vdb = test_data.Bunny().vdb
        self._test_leaf_nodes(grid=bunny_vdb)
        #  self._test_to_vdb(grid=bunny_vdb)
        self._test_voxel_size(grid=bunny_vdb)
        self._test_get_item(grid=bunny_vdb)
        self._test_background_value(grid=bunny_vdb)
        self._test_leaf_node_shape(grid=bunny_vdb)

    def test_torus(self):
        torus_vdb = test_data.Torus().vdb
        self._test_leaf_nodes(grid=torus_vdb)
        self._test_to_vdb(grid=torus_vdb)
        self._test_voxel_size(grid=torus_vdb)
        self._test_get_item(grid=torus_vdb)
        self._test_background_value(grid=torus_vdb)
        self._test_leaf_node_shape(grid=torus_vdb)

    def test_sphere(self):
        sphere_vdb = vdb.createLevelSetSphere(2.0, voxelSize=0.1)
        self._test_leaf_nodes(grid=sphere_vdb)
        self._test_to_vdb(grid=sphere_vdb)
        self._test_voxel_size(grid=sphere_vdb)
        self._test_get_item(grid=sphere_vdb)
        self._test_background_value(grid=sphere_vdb)
        self._test_leaf_node_shape(grid=sphere_vdb)

    def test_empty_grid(self):
        self._test_leaf_nodes(grid=vdb.FloatGrid())
        self._test_voxel_size(grid=vdb.FloatGrid())

    def test_bool_grid_conversion(self):
        # This type is not supported
        with self.assertRaises(ValueError):
            LeafNodeGrid(vdb.BoolGrid())

    def test_vec3sgrid_conversion(self):
        # This type is not supported
        with self.assertRaises(ValueError):
            LeafNodeGrid(vdb.Vec3SGrid)

    def test_leaf_node_normalization(self):
        grid = test_data.Bunny().vdb

        # Test no normalization first
        float_grid = LeafNodeGrid(grid, normalize=False)
        _, nodes_a = float_grid.numpy()
        self.assertAlmostEqual(nodes_a.max(), +float_grid.background)
        self.assertAlmostEqual(nodes_a.min(), -float_grid.background)

        # Now normalize and check for the [-1, +1] range
        normalized_grid = LeafNodeGrid(grid, normalize=True)
        _, normalized_nodes_a = normalized_grid.numpy()
        self.assertAlmostEqual(normalized_nodes_a.max(), +np.float32(1.0))
        self.assertAlmostEqual(normalized_nodes_a.min(), -np.float32(1.0))


if __name__ == "__main__":
    unittest.main()
