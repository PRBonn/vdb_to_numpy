import numpy as np
import pyopenvdb as vdb


def level_set_to_numpy(grid: vdb.FloatGrid) -> np.ndarray:
    shape = grid.evalActiveVoxelDim()
    start = grid.evalActiveVoxelBoundingBox()[0]
    sdf_volume = np.zeros(shape, dtype=np.float32)
    grid.copyToArray(sdf_volume, ijk=start)
    return sdf_volume


def compare_2_grids(grid1, grid2):
    """Extracts a 1d array representation of ALL values of a given VDB grid and
    checks for equality on a per-value base comparission."""

    def level_set_to_1d(grid):
        sdf = []
        for f in grid.iterAllValues():
            sdf.append(f.value)
        return np.asarray(sdf, dtype=np.float32)

    return (level_set_to_1d(grid1) == level_set_to_1d(grid2)).all()


# Input VDB, a simple sphere
vdb_grid_in = vdb.createLevelSetSphere(1.0, voxelSize=0.1)
vdb_grid_in.name = "vdb_grid in"

# Covert to dense np array
np_grid = level_set_to_numpy(vdb_grid_in)

# Covert back to a VDB grid
vdb_grid_out = vdb.FloatGrid()
vdb_grid_out.transform = vdb_grid_in.transform
vdb_grid_out.background = vdb_grid_in.background
vdb_grid_out.gridClass = vdb_grid_in.gridClass
vdb_grid_out.name = "vdb_grid out"
vdb_grid_out.copyFromArray(np_grid, ijk=vdb_grid_in.evalActiveVoxelBoundingBox()[0])

print("in == out (1D representation)?", compare_2_grids(vdb_grid_in, vdb_grid_out))
print("vdb_grid_in.activeLeafVoxelCount()", vdb_grid_in.activeLeafVoxelCount())
print("vdb_grid_out.activeLeafVoxelCount()", vdb_grid_out.activeLeafVoxelCount())
print("Same background?", vdb_grid_in.background == vdb_grid_out.background)
print("Same min value ?", vdb_grid_in.evalMinMax()[0] == vdb_grid_out.evalMinMax()[0])
print("Same max value ?", vdb_grid_in.evalMinMax()[1] == vdb_grid_out.evalMinMax()[1])
vdb.write("in.vdb", vdb_grid_in)
vdb.write("out.vdb", vdb_grid_out)
