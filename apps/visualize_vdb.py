#!/usr/bin/env python3
# coding: utf-8
import os

import click
import open3d as o3d

from vdb_to_numpy.utils import preprocess_mesh
from vdb_to_numpy.vdb_tools import mesh_to_level_set, visualize_vdb_grid


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--voxel_size", type=float, default=0.5)
@click.option("--scale", is_flag=True, default=False)
def main(filename, voxel_size, scale):
    """Open a mesh, converting to levelset and visualize using OpenVDB."""
    filename = os.path.abspath(filename)
    mesh = o3d.io.read_triangle_mesh(filename)

    try:
        print("Preprocessing input mesh...")
        mesh = preprocess_mesh(mesh, scale=scale)
    except ValueError:
        print("Could not preprocess_mesh {}".format(filename))
        pass

    # Convert it to a level set using OpenVDB tools
    print("Converting Triangle Mesh to a level set volume...")
    vdb_grid = mesh_to_level_set(mesh, voxel_size)
    visualize_vdb_grid(grid=vdb_grid, filename=filename, verbose=False)


if __name__ == "__main__":
    main()
