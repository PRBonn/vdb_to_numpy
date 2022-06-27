import pyopenvdb as vdb

from ..pybind import vdb_pybind


def blend_grids(grid_a: vdb.FloatGrid, grid_b: vdb.FloatGrid, eta: float = 0.9) -> None:
    """blend 2 vdb grids."""
    vdb_pybind._blend_grids(grid_a, grid_b, eta)


def normalize_grid(grid: vdb.FloatGrid) -> vdb.FloatGrid:
    """Normalize VDB grid."""
    return vdb_pybind._normalize_grid(grid)
