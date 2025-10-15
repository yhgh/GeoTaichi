"""GeoWarp compatibility package exposing Warp backend hooks and I/O shims."""

from .backend import init
from .dem import DEMSystem, MeshBoundary, PeriodicBoundary, VolumeBoundary
from .geom import SignedDistanceVolume, TriangleMesh
from .io_vtk import write_points_vtu, write_grid_vts
from .mpm import ExplicitMPMSolver, ImplicitMPMSolver

__all__ = [
    "init",
    "write_points_vtu",
    "write_grid_vts",
    "DEMSystem",
    "MeshBoundary",
    "VolumeBoundary",
    "PeriodicBoundary",
    "TriangleMesh",
    "SignedDistanceVolume",
    "ExplicitMPMSolver",
    "ImplicitMPMSolver",
]
