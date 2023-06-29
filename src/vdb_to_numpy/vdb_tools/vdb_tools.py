import os
from typing import Tuple

import numpy as np
import open3d as o3d
import pyopenvdb as vdb


def mesh_to_level_set(mesh, voxel_size, half_width=3):
    return vdb.FloatGrid.createLevelSetFromPolygons(
        points=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles),
        transform=vdb.createLinearTransform(voxelSize=voxel_size),
        halfWidth=half_width,
    )


def level_set_to_triangle_mesh(grid):
    points, quads = grid.convertToQuads()
    faces = np.array([[[f[0], f[1], f[2]], [f[0], f[2], f[3]]] for f in quads]).reshape((-1, 3))
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(points),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    return mesh


def level_set_to_numpy(grid: vdb.FloatGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Given an input level set (in vdb format) extract the dense array representation of the volume
    and convert it to a numpy array.

    You could check the output of the numpy array by running marching cubes over the
    volume and extracting the mesh for visualization.
    """
    # Dimensions of the axis-aligned bounding box of all active voxels.
    shape = grid.evalActiveVoxelDim()
    # Return the coordinates of opposite corners of the axis-aligned bounding
    # box of all active voxels.
    start = grid.evalActiveVoxelBoundingBox()[0]
    # Create a dense array of zeros
    sdf_volume = np.zeros(shape, dtype=np.float32)
    # Copy the volume to the output, starting from the first occupied voxel
    grid.copyToArray(sdf_volume, ijk=start)
    # solve background error see OpenVDB#1096
    sdf_volume[sdf_volume < grid.evalMinMax()[0]] = grid.background

    # In order to put a mesh back into its original coordinate frame we also
    # need to know where the volume was located
    origin_xyz = grid.transform.indexToWorld(start)
    return sdf_volume, origin_xyz


def visualize_vdb_grid(grid, filename, verbose=True):
    # Save it to file in /tmp
    grid_name = os.path.split(filename.split(".")[0])[-1]
    grid_fn = os.path.join("/tmp", grid_name + ".vdb")
    vdb.write(grid_fn, grid)

    # Plot the results
    os.system("vdb_print -l {}".format(grid_fn)) if verbose else None
    os.system("vdb_view {}".format(grid_fn))
