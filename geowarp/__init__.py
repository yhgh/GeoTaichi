"""GeoWarp compatibility package exposing Warp backend hooks and I/O shims."""

from .backend import init
from .dem import (
    ContactPair,
    DEMMaterial,
    DEMMaterialStruct,
    DEMSystem,
    MeshBoundary,
    PeriodicBoundary,
    VolumeBoundary,
    combine_materials,
    coulomb_friction,
    relative_velocity,
    spring_dashpot_contact,
)
from .geom import SignedDistanceVolume, TriangleMesh
from .io_vtk import write_points_vtu, write_grid_vts
from .mpm import (
    DruckerPrager,
    DruckerPragerStruct,
    ExplicitMPMSolver,
    ImplicitMPMSolver,
    LinearElasticParams,
    MaterialState,
    MohrCoulomb,
    MohrCoulombStruct,
    elastic_trial_stress,
    radial_return_drucker_prager,
    radial_return_mohr_coulomb,
)
from .mpdem import MPDEMCoupling
from .utils import (
    EnergyAccumulator,
    compute_contact_energy,
    compute_kinetic_energy,
    compute_potential_energy,
)

__all__ = [
    "init",
    "write_points_vtu",
    "write_grid_vts",
    "DEMSystem",
    "DEMMaterial",
    "DEMMaterialStruct",
    "ContactPair",
    "combine_materials",
    "relative_velocity",
    "spring_dashpot_contact",
    "coulomb_friction",
    "MeshBoundary",
    "VolumeBoundary",
    "PeriodicBoundary",
    "TriangleMesh",
    "SignedDistanceVolume",
    "MaterialState",
    "LinearElasticParams",
    "elastic_trial_stress",
    "DruckerPrager",
    "DruckerPragerStruct",
    "ExplicitMPMSolver",
    "ImplicitMPMSolver",
    "MohrCoulomb",
    "MohrCoulombStruct",
    "radial_return_drucker_prager",
    "radial_return_mohr_coulomb",
    "MPDEMCoupling",
    "EnergyAccumulator",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "compute_contact_energy",
]
