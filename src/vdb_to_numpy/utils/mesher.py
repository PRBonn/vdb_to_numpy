import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes


def extract_mesh(volume, mask=None):
    """Run marching_cubes and extract a triangular mesh of the volume.

    Parameters - copied from skimage -
    ----------
    volume : (M, N, P) array
        Input data volume to find isosurfaces. Will internally be
    mask : (M, N, P) array
        Boolean array. The marching cube algorithm will be computed only on
        True elements. This will save computational time when interfaces
        are located within certain region of the volume M, N, P-e.g. the top
        half of the cube-and also allow to compute finite surfaces-i.e. open
        surfaces that do not end at the border of the cube.
        converted to float32 if necessary.
    """
    vertices, faces, _, _ = marching_cubes(volume, level=0, mask=mask)
    vertices = o3d.utility.Vector3dVector(vertices)
    triangles = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
    mesh.compute_vertex_normals()
    return mesh
