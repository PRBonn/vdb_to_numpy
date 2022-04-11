#!/usr/bin/env python3
import os

import click
import open3d as o3d

from vdb_to_numpy.utils import scale_to_unit_sphere as scale_mesh


@click.command()
@click.option("--in_dir", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(exists=False), required=True)
@click.option("--padding", type=float, default=0.1)
def main(in_dir, out_dir, padding):
    """Scale a set of meshes using Open3D.

    This tool is inspired by:
    https://github.com/davidstutz/mesh-fusion/blob/master/1_scale.py but we
    make use of standard tools like Open3D and thus reduce 700 lines of code to
    roughly 50.
    """
    # Make sure out_dir exists
    os.makedirs(out_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            print("Scaling {}...".format(filename))
            mesh = o3d.io.read_triangle_mesh(os.path.join(dirpath, filename))
            mesh = scale_mesh(mesh, padding)
            o3d.io.write_triangle_mesh(os.path.join(out_dir, filename), mesh)


if __name__ == "__main__":
    main()
