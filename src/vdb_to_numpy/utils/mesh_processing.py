import copy

import manifold
import numpy as np
import open3d as o3d


def scale_to_unit_sphere(mesh, scale=1, padding=0.1):
    """Scale the input mesh into a unit sphere."""
    # Get bbox of original mesh
    bbox = mesh.get_axis_aligned_bounding_box()

    # Translate the mesh
    scaled_mesh = copy.deepcopy(mesh)
    scaled_mesh.translate(-bbox.get_center())
    distances = np.linalg.norm(np.asarray(scaled_mesh.vertices), axis=1)
    scaled_mesh.scale(1 / np.max(distances), center=[0, 0, 0])
    scaled_mesh.scale(scale * (1 - padding), center=[0, 0, 0])
    return scaled_mesh


def scale_to_unit_cube(mesh, scale=1, padding=0.1):
    """Scale the input mesh into a unit cube."""
    # Get bbox of original mesh
    bbox = mesh.get_axis_aligned_bounding_box()

    # Translate the mesh
    scaled_mesh = copy.deepcopy(mesh)
    scaled_mesh.translate(-bbox.get_center())
    scaled_mesh.scale(2 / bbox.get_max_extent(), center=[0, 0, 0])
    scaled_mesh.scale(scale * (1 - padding), center=[0, 0, 0])
    return scaled_mesh


def watertight_mesh(mesh, depth=8):
    """Conver the input mesh to a watertight model."""
    processor = manifold.Processor(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
    )

    output_vertices, output_triangles = processor.get_manifold_mesh(depth)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(output_vertices),
        o3d.utility.Vector3iVector(output_triangles),
    )


def preprocess_mesh(mesh, scale=False, watertight=False):
    """The mesh MUST be a closed surface, but not necessary watertight and can
    also contain self-intersecting faces, in contrast to most of mesh-to-sdf
    algorithms.

    Scaling is not mandatory, but it's for your own sanity
    """
    mesh = scale_to_unit_sphere(mesh) if scale else mesh
    mesh = watertight_mesh(mesh) if watertight else mesh
    mesh.compute_vertex_normals()
    return mesh
