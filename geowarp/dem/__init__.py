"""Discrete element method support for GeoWarp."""

from .boundary import MeshBoundary, PeriodicBoundary, VolumeBoundary
from .core import DEMSystem
from .materials import (
    ContactPair,
    DEMMaterial,
    DEMMaterialStruct,
    combine_materials,
    coulomb_friction,
    relative_velocity,
    spring_dashpot_contact,
)

__all__ = [
    "DEMSystem",
    "MeshBoundary",
    "VolumeBoundary",
    "PeriodicBoundary",
    "DEMMaterial",
    "DEMMaterialStruct",
    "ContactPair",
    "combine_materials",
    "relative_velocity",
    "spring_dashpot_contact",
    "coulomb_friction",
]
