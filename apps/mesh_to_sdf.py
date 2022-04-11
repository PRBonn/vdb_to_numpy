#!/usr/bin/env python3
# coding: utf-8
import os

import click
import numpy as np
import open3d as o3d

from vdb_to_numpy.utils import extract_mesh, preprocess_mesh
from vdb_to_numpy.vdb_tools import (
    level_set_to_numpy,
    mesh_to_level_set,
    visualize_vdb_grid,
)


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--voxel_size",
    type=float,
    default=0.01,
    help="Voxel size of the resulting SDF volume",
)
@click.option(
    "--scale",
    is_flag=True,
    default=False,
    help="Fit the input mesh into a unit sphere before converting it to a SDF",
)
@click.option(
    "--watertight",
    is_flag=True,
    default=False,
    help="Convert the input model into a watertight one before computing the SDF",
)
@click.option(
    "--visualize",
    is_flag=True,
    default=False,
    help="When running it locally, visualize the pipeline",
)
@click.option(
    "--mcubes",
    is_flag=True,
    default=False,
    help="Run marching cubes on the output SDF and store the mesh for inspection",
)
def main(filename, voxel_size, watertight, scale, mcubes, visualize):
    """Convert triangular meshes into dense SDF(Singed distance field) volumes in numpy format.

    The input to the script is any triangular mesh in any supported
    format by Open3D (.ply, .obj, .off, etc...). The script will convert
    the input mesh to a VDB grid representation and then extract its
    dense representation as numpy array of shape (X, Y, Z) of type
    np.float64. The surface is represented in this array as the
    0-isosurface and can be extracted by running marching cubes. Use the
    --mcubes flag to inspect this results.

    You typically want to use the ``--scale`` and ``--watertight`` flags to
    make sure you can robustly extract the SDF representation from the mesh.
    """
    filename = os.path.abspath(filename)
    file_extension = os.path.splitext(filename)[-1]
    model_name = os.path.splitext(filename)[0]

    # When using trimesh use degenerate_triangles
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh]) if visualize else None

    print("Preprocessing input mesh...")
    mesh = preprocess_mesh(mesh, scale=scale, watertight=watertight)

    # Convert it to a level set using OpenVDB tools
    print("Converting Triangle Mesh to a level set volume...")
    vdb_grid = mesh_to_level_set(mesh, voxel_size)

    visualize_vdb_grid(vdb_grid, filename) if visualize else None

    # Convert the VDB grid to a dense numpy array
    print("Converting to a dense SDF representation")
    sdf_volume, _ = level_set_to_numpy(vdb_grid)
    print("Volume output:")
    print("sdf_volume.shape = ", sdf_volume.shape)
    print("sdf_volume.min() = ", sdf_volume.min())
    print("sdf_volume.max() = ", sdf_volume.max())

    # You can now save the sdf_volume as a np array and read it later on
    numpy_filename = model_name + "_sdf.npy"
    print("Saving sdf_volume to", numpy_filename)
    np.save(numpy_filename, sdf_volume)

    if mcubes:
        print("Meshing dense volume by running marching cubes")
        sdf_mesh = extract_mesh(sdf_volume)
        o3d.visualization.draw_geometries([sdf_mesh]) if visualize else None
        mesh_filename = model_name + "_sdf_mesh" + file_extension
        print("Saving sdf_volume mesh to", mesh_filename)
        o3d.io.write_triangle_mesh(mesh_filename, sdf_mesh)


if __name__ == "__main__":
    main()
