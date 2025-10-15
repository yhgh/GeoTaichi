"""DEM material parameter containers and helper kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import warp as wp


@dataclass
class DEMMaterial:
    """Particle material definition mirroring the legacy GeoTaichi fields."""

    density: float
    kn: float
    kt: float
    restitution: float = 0.0
    friction: float = 0.5
    rolling_friction: float = 0.0
    damping: float = 0.0
    cohesion: float = 0.0

    def as_struct(self) -> "DEMMaterialStruct":
        return DEMMaterialStruct(
            density=self.density,
            kn=self.kn,
            kt=self.kt,
            restitution=self.restitution,
            friction=self.friction,
            rolling_friction=self.rolling_friction,
            damping=self.damping,
            cohesion=self.cohesion,
        )


@wp.struct
class DEMMaterialStruct:
    density: float
    kn: float
    kt: float
    restitution: float
    friction: float
    rolling_friction: float
    damping: float
    cohesion: float


@wp.struct
class ContactPair:
    normal_stiffness: float
    tangential_stiffness: float
    damping: float
    friction: float
    rolling_friction: float
    cohesion: float


def combine_materials(a: DEMMaterial, b: DEMMaterial) -> ContactPair:
    """Return effective contact parameters for a pair of materials."""

    kn = 2.0 * a.kn * b.kn / (a.kn + b.kn)
    kt = 2.0 * a.kt * b.kt / (a.kt + b.kt)
    damping = max(a.damping, b.damping)
    friction = min(a.friction, b.friction)
    rolling = min(a.rolling_friction, b.rolling_friction)
    cohesion = min(a.cohesion, b.cohesion)
    return ContactPair(
        normal_stiffness=kn,
        tangential_stiffness=kt,
        damping=damping,
        friction=friction,
        rolling_friction=rolling,
        cohesion=cohesion,
    )


@wp.func
def relative_velocity(
    vi: wp.vec3f,
    wi: wp.vec3f,
    vj: wp.vec3f,
    wj: wp.vec3f,
    contact_vector: wp.vec3f,
    radius_i: float,
    radius_j: float,
) -> wp.vec3f:
    ri = contact_vector * (radius_i / (radius_i + radius_j + 1.0e-6))
    rj = -contact_vector + ri
    vel_i = vi + wp.cross(wi, ri)
    vel_j = vj + wp.cross(wj, rj)
    return vel_i - vel_j


@wp.func
def spring_dashpot_contact(
    delta: wp.vec3f,
    dv: wp.vec3f,
    normal: wp.vec3f,
    pair: ContactPair,
) -> Tuple[wp.vec3f, float, float]:
    overlap = -wp.dot(delta, normal)
    fn = wp.vec3f(0.0, 0.0, 0.0)
    normal_energy = 0.0
    damping_energy = 0.0

    if overlap > 0.0:
        normal_force = pair.normal_stiffness * overlap
        damping_force = pair.damping * wp.dot(dv, normal)
        normal_force -= damping_force
        normal_force += pair.cohesion
        fn = normal_force * normal
        normal_energy = 0.5 * pair.normal_stiffness * overlap * overlap
        damping_energy = 0.5 * pair.damping * damping_force * damping_force

    return fn, normal_energy, damping_energy


@wp.func
def coulomb_friction(
    dv: wp.vec3f,
    normal: wp.vec3f,
    normal_force: wp.vec3f,
    pair: ContactPair,
) -> Tuple[wp.vec3f, float]:
    tangent = dv - wp.dot(dv, normal) * normal
    tangent_mag = wp.length(tangent)
    ft = wp.vec3f(0.0, 0.0, 0.0)
    slip_energy = 0.0

    if tangent_mag > 1.0e-6:
        direction = tangent / tangent_mag
        limit = pair.friction * wp.length(normal_force)
        magnitude = wp.min(limit, pair.tangential_stiffness * tangent_mag)
        ft = -magnitude * direction
        slip_energy = magnitude * tangent_mag

    return ft, slip_energy


__all__ = [
    "DEMMaterial",
    "DEMMaterialStruct",
    "ContactPair",
    "combine_materials",
    "relative_velocity",
    "spring_dashpot_contact",
    "coulomb_friction",
]

