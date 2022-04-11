import numpy as np
import open3d as o3d
import pyopenvdb as vdb

from ..pybind import vdb_pybind


def vdb_to_triangle_mesh(vdb_grid: vdb.FloatGrid):
    """Returns an Open3D TriangleMesh format, maybe we should just return the triangles and vertices
    and let the user decide what to do."""
    if not isinstance(vdb_grid, vdb.FloatGrid):
        raise ValueError("GridType: '{}' not supported".format(type(vdb_grid)))
    voxel_size = np.float32(vdb_grid.transform.voxelSize()[0])
    vertices, triangles = vdb_pybind._extract_triangle_mesh(vdb_grid, voxel_size)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    return mesh
