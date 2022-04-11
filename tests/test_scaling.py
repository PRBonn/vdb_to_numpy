"""Test the Mesh scaling functions."""
import copy
import unittest

import numpy as np
import numpy.testing
import open3d as o3d

from vdb_to_numpy.utils import scale_to_unit_sphere, scale_to_unit_cube


# TODO: complete the unit tests to check for unit_cube the same way we do for
# unit_sphere
class MeshScalingTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Validates that the scaling function of to unit sphere is working."""
        # Let's start creating a reference mesh, centered at (0, 0, 0)
        # Unit Sphere
        self.unit_radius = 1.0
        self.unit_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        self.unit_sphere_volume = self._sphere_volume(self.unit_radius)
        self.assertAlmostEqual(
            self.unit_sphere.get_volume(),
            self.unit_sphere_volume,
            delta=0.1 * self.unit_sphere_volume,
        )
        # Unit Cube
        self.unit_cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        self.unit_cube_volume = self._cube_volume(width=1.0, height=1.0, depth=1.0)
        self.assertAlmostEqual(
            self.unit_cube.get_volume(),
            self.unit_cube_volume,
            delta=0.1 * self.unit_cube_volume,
        )

    def test_unit_sphere_fitting(self):
        radius, mesh = self.sample_random_sphere(max_radius=100)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        scaled_volume = self._sphere_volume(radius)
        self.assertAlmostEqual(
            mesh.get_volume(),
            scaled_volume,
            delta=0.1 * scaled_volume,
        )

        # Now fit the mesh back into a unit sphere and test the results
        scaled_mesh = scale_to_unit_sphere(mesh, scale=1, padding=0)
        self.assertAlmostEqual(
            scaled_mesh.get_volume(),
            self.unit_sphere_volume,
            delta=0.1 * self.unit_sphere_volume,
        )

    def test_mesh_scaling(self):
        radius, mesh = self.sample_random_sphere(max_radius=100)
        # After fitting the mesh to the unit sphere, we can pick a new scale
        scale = int(100 * np.random.random_sample() + 1)
        # Sphere fitting
        scaled_mesh = scale_to_unit_sphere(mesh, scale=scale, padding=0)
        unit_volume_scaled = self._sphere_volume(scale * self.unit_radius)
        self.assertAlmostEqual(
            scaled_mesh.get_volume(),
            unit_volume_scaled,
            delta=0.1 * unit_volume_scaled,
        )

        # Cube fitting
        scaled_mesh = scale_to_unit_cube(mesh, scale=scale, padding=0)
        unit_volume_scaled = self._sphere_volume(scale * self.unit_radius)
        self.assertAlmostEqual(
            scaled_mesh.get_volume(),
            unit_volume_scaled,
            delta=0.1 * unit_volume_scaled,
        )

    def test_mesh_scaling_padding(self):
        radius, mesh = self.sample_random_sphere(max_radius=100)

        # Pick a random padding value between [0, 1)
        padding = np.random.random_sample()
        # Scale the mest now using padding
        scaled_mesh = scale_to_unit_sphere(mesh, scale=1, padding=padding)
        # if padding is 0.1, this means that the radius is now 0.9
        unit_volume_padding = self._sphere_volume((1 - padding) * self.unit_radius)
        self.assertAlmostEqual(
            scaled_mesh.get_volume(),
            unit_volume_padding,
            delta=0.1 * unit_volume_padding,
        )

    def test_mesh_scaling_center(self):
        # Random sphere should always be centerd at zero
        radius, mesh = self.sample_random_sphere(max_radius=100)
        mesh_center = mesh.get_axis_aligned_bounding_box().get_center()
        numpy.testing.assert_array_almost_equal(mesh_center, np.zeros(3))

        # Sample a random translation vector
        translation = 100 * np.random.random_sample(size=3)
        mesh.translate(translation)
        mesh_center_t = mesh.get_axis_aligned_bounding_box().get_center()
        numpy.testing.assert_array_almost_equal(mesh_center_t, translation)

        # After fitting to a unit sphere, the center should be back to the zero
        scaled_mesh = scale_to_unit_sphere(mesh, scale=1, padding=0)
        scaled_center = scaled_mesh.get_axis_aligned_bounding_box().get_center()
        numpy.testing.assert_array_almost_equal(scaled_center, np.zeros(3))

        # Check that the volume didn't change
        self.assertAlmostEqual(
            scaled_mesh.get_volume(),
            self.unit_sphere_volume,
            delta=0.1 * self.unit_sphere_volume,
        )

    @staticmethod
    def sample_random_sphere(max_radius):
        """Sample a random sphere from a normal distribution."""
        radius = max_radius * np.random.random_sample()
        return radius, o3d.geometry.TriangleMesh.create_sphere(radius=radius)

    @staticmethod
    def _sphere_volume(radius):
        """Computes the volume of a sphere."""
        return (4 / 3) * np.pi * radius**3

    @staticmethod
    def _cube_volume(width, height, depth):
        """Computes the volume of a cube."""
        return width * height * depth
