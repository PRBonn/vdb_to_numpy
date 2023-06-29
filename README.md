# VDB to Numpy

A python utility library to convert Triangular Meshes/VDB grids into numpy arrays. This project was
created to work with [vdbfusion](https://github.com/PRBonn/vdbfusion)

## Dependencies

- OpenVDB, please check official documentation.
- Check the [Dockerfile](./docker/builder/Dockerfile) for more details ...

## Usage

### With VDBFusion

The main idea of this `vdb_to_numpy` package was to use it with maps generated with
[vdbfusion](https://github.com/PRBonn/vdbfusion). More specifically, to train some neural networks
with this type of data.

To use this package in such way, you first need some VDBs, to do so, go and checkout the [vdbfusion
examples](https://github.com/PRBonn/vdbfusion/tree/main/examples/python). This examples will spit
some VDB files, that you can right away use with some of the [apps](./apps) on this project. There
are some other [experiments](./experiments) to checkout.

### Mesh-to-sdf

If you only want to use this package to convert triangular meshes to SDF fields then you probably
want to use the `mesh_to_sdf` docker container to convert your meshes and then runaway. If you need
extra funcionallity you can clone this repo and install the tool locally and start messing around
with the example [apps](./apps/). If not, this is the easiest entry point:

For doing so, just run this command, your current working directory will be
mounted to the `/models/` path in the container.

```sh
docker run -it --rm \
    -v $(pwd):/models \
    --user 1000:1000 \
    ignaciovizzo/vdb_to_numpy:mesh_to_sdf \
    /models/tests/test_data/bunny.ply \
    --scale \
    --watertight \
    --mcubes
```

This command mounts your current working directory to the /models directory in
the docker container and executes the [mesh_to_sdf.py](apps/mesh_to_sdf.py)
script on the mesh file `bunny.ply` in the current directory.

The output of this command will be the following files:

```sh
├── tests
│   ├── test_data
│   │   ├── bunny.ply           # Input mesh, not watertigh, not to scale
│   │   ├── bunny_sdf.npy       # Output numpy SDF dense grid
│   │   ├── bunny_sdf_mesh.ply  # Output mesh, after running marching cubes
```

If you need extra help just:

```sh
docker run -it --rm ignaciovizzo/vdb_to_numpy:mesh_to_sdf --help
```

**NOTE:** I've created this repoisotry in 2021 and I'm currently not actively using it. The API is
[tested](./tests) but use it at your own risk ;)
