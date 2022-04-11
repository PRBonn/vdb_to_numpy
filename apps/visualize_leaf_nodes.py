#!/usr/bin/env python3
# coding: utf-8
import os

import click
from diskcache import FanoutCache
import open3d as o3d
import pyopenvdb as vdb

from vdb_to_numpy import LeafNodeGrid
from vdb_to_numpy.utils import SerializableMesh, preprocess_mesh
from vdb_to_numpy.vdb_tools import (
    level_set_to_triangle_mesh,
    mesh_to_level_set,
    visualize_vdb_grid,
)
from vdb_to_numpy.visualization import LeafNodeGridVisualizer

cache = FanoutCache(
    directory="cache/",
    shards=64,
    timeout=1,
)


def _get_leaf_node_grid(filename, voxel_size, scale, watertight, visualize_vdb):
    file_extension = os.path.splitext(filename)[-1]
    # If we alredy got a VDB grid, then skip the mesh-to-volume step
    if file_extension == ".vdb":
        print("Converting level set volume to Triangle Mesh...")
        mesh = level_set_to_triangle_mesh(vdb.readAll(filename)[0][0])
    else:
        # When using trimesh use degenerate_triangles
        mesh = o3d.io.read_triangle_mesh(filename)

    # Prerocess the mesh
    print("Preprocessing input mesh...")
    mesh = preprocess_mesh(mesh, scale=scale, watertight=watertight)

    # Convert it to a level set using OpenVDB tools
    print("Converting Triangle Mesh to a level set volume...")
    vdb_grid = mesh_to_level_set(mesh, voxel_size)

    visualize_vdb_grid(vdb_grid, filename) if visualize_vdb else None

    # Convert the VDB grid to a numpy-based LeafNodeGrid object
    return LeafNodeGrid(vdb_grid), SerializableMesh(mesh)


def get_leaf_node_grid(filename, voxel_size, scale, watertight, visualize_vdb, no_cache):
    if no_cache:
        f = _get_leaf_node_grid
    else:
        print("[WARNING] Reading data from cache")
        f = cache.memoize(typed=True)(_get_leaf_node_grid)
    grid, _mesh = f(filename, voxel_size, scale, watertight, visualize_vdb)
    return grid, _mesh.as_open3d()


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
    "--no-cache",
    is_flag=True,
    default=False,
    help="Don't read data from the local cache",
)
@click.option("--visualize_vdb", is_flag=True, default=False)
def main(filename, voxel_size, scale, watertight, visualize_vdb, no_cache):
    filename = os.path.abspath(filename)
    # Convert the VDB grid to a numpy-based LeafNodeGrid object
    grid, mesh = get_leaf_node_grid(
        filename, voxel_size, scale, watertight, visualize_vdb, no_cache
    )
    vis = LeafNodeGridVisualizer(grid, mesh)
    vis.set_render_options(
        mesh_show_wireframe=True,
        mesh_show_back_face=True,
    )
    vis.run()


if __name__ == "__main__":
    main()
