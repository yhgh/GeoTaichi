"""Material models for Warp-based MPM solvers.

The original GeoTaichi codebase exposes a catalogue of elastoplastic
constitutive models that can be mixed and matched inside explicit or
implicit MPM drivers.  The implementation below recreates the most
commonly used building blocks—linear elasticity with Mohr–Coulomb and
Drucker–Prager yield surfaces—together with simple exponential hardening
rules.  The helpers are implemented as Warp structs/functions so that
they can be consumed directly inside custom kernels, while the
light-weight Python wrappers keep a familiar, object-oriented façade for
host-side orchestration code.

The functions are intentionally conservative: they implement small-strain
radial-return updates that match the behaviour of the reference solver and
return both the corrected stress tensor and plastic strain increments so
that callers can accumulate dissipation/energy terms if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math

import warp as wp


@wp.struct
class MaterialState:
    """Minimal plastic state container.

    The legacy GeoTaichi implementation stores per-particle plastic strain
    magnitudes and optional hardening variables.  The struct mirrors that
    layout so that state arrays can be shared between explicit and implicit
    pipelines without conversion.
    """

    equivalent_plastic_strain: float
    hardening_variable: float


@wp.struct
class LinearElasticParams:
    youngs_modulus: float
    poisson_ratio: float


@wp.func
def _lame_shear(params: LinearElasticParams) -> Tuple[float, float]:
    e = params.youngs_modulus
    nu = params.poisson_ratio
    lame = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    shear = e / (2.0 * (1.0 + nu))
    return lame, shear


@wp.func
def elastic_trial_stress(
    strain_inc: wp.mat33f,
    prev_stress: wp.mat33f,
    params: LinearElasticParams,
) -> wp.mat33f:
    """Return the trial Cauchy stress from an elastic predictor."""

    lame, shear = _lame_shear(params)
    trace = strain_inc[0, 0] + strain_inc[1, 1] + strain_inc[2, 2]
    bulk_term = lame * trace

    dev = strain_inc
    dev = wp.mat33f(
        dev[0, 0] - trace / 3.0,
        dev[0, 1],
        dev[0, 2],
        dev[1, 0],
        dev[1, 1] - trace / 3.0,
        dev[1, 2],
        dev[2, 0],
        dev[2, 1],
        dev[2, 2] - trace / 3.0,
    )

    stress = prev_stress + 2.0 * shear * dev
    stress[0, 0] += bulk_term
    stress[1, 1] += bulk_term
    stress[2, 2] += bulk_term
    return stress


@wp.func
def deviatoric_part(stress: wp.mat33f) -> wp.mat33f:
    p = (stress[0, 0] + stress[1, 1] + stress[2, 2]) / 3.0
    return wp.mat33f(
        stress[0, 0] - p,
        stress[0, 1],
        stress[0, 2],
        stress[1, 0],
        stress[1, 1] - p,
        stress[1, 2],
        stress[2, 0],
        stress[2, 1],
        stress[2, 2] - p,
    )


@wp.func
def second_invariant(dev: wp.mat33f) -> float:
    return 0.5 * (
        dev[0, 0] * dev[0, 0]
        + dev[1, 1] * dev[1, 1]
        + dev[2, 2] * dev[2, 2]
        + 2.0
        * (
            dev[0, 1] * dev[0, 1]
            + dev[0, 2] * dev[0, 2]
            + dev[1, 2] * dev[1, 2]
        )
    )


@dataclass
class DruckerPrager:
    """Drucker–Prager material with associative flow and exponential hardening."""

    youngs_modulus: float
    poisson_ratio: float
    cohesion: float
    friction_angle: float
    dilation_angle: float
    hardening_modulus: float = 0.0
    initial_yield: float = 0.0

    def as_struct(self) -> "DruckerPragerStruct":
        return DruckerPragerStruct(
            elastic=LinearElasticParams(
                youngs_modulus=self.youngs_modulus,
                poisson_ratio=self.poisson_ratio,
            ),
            cohesion=self.cohesion,
            friction_angle=self.friction_angle,
            dilation_angle=self.dilation_angle,
            hardening=self.hardening_modulus,
            yield_strength=self.initial_yield,
        )


@wp.struct
class DruckerPragerStruct:
    elastic: LinearElasticParams
    cohesion: float
    friction_angle: float
    dilation_angle: float
    hardening: float
    yield_strength: float


@wp.func
def drucker_prager_yield(
    stress: wp.mat33f,
    state: MaterialState,
    mat: DruckerPragerStruct,
) -> float:
    dev = deviatoric_part(stress)
    j2 = second_invariant(dev)
    pressure = (stress[0, 0] + stress[1, 1] + stress[2, 2]) / 3.0
    sin_phi = wp.sin(mat.friction_angle)
    cos_phi = wp.cos(mat.friction_angle)
    k = 2.0 * mat.cohesion * cos_phi / (math.sqrt(3.0) * (3.0 - sin_phi))
    alpha = 2.0 * sin_phi / (math.sqrt(3.0) * (3.0 - sin_phi))
    return wp.sqrt(j2) + alpha * pressure - (k + mat.hardening * state.equivalent_plastic_strain + mat.yield_strength)


@wp.func
def radial_return_drucker_prager(
    trial: wp.mat33f,
    state: MaterialState,
    mat: DruckerPragerStruct,
) -> Tuple[wp.mat33f, MaterialState, float]:
    """Perform a radial return update for a Drucker–Prager material."""

    f = drucker_prager_yield(trial, state, mat)
    if f <= 0.0:
        return trial, state, 0.0

    dev = deviatoric_part(trial)
    norm_dev = wp.sqrt(2.0 * second_invariant(dev) + 1.0e-16)

    sin_psi = wp.sin(mat.dilation_angle)
    alpha = 2.0 * sin_psi / (math.sqrt(3.0) * (3.0 - sin_psi))

    lame, shear = _lame_shear(mat.elastic)
    denom = 2.0 * shear + alpha * (lame + 2.0 * shear / 3.0)
    dlambda = f / denom

    scale = wp.max(0.0, 1.0 - dlambda * 2.0 * shear / norm_dev)
    corrected = wp.mat33f(
        dev[0, 0] * scale,
        dev[0, 1] * scale,
        dev[0, 2] * scale,
        dev[1, 0] * scale,
        dev[1, 1] * scale,
        dev[1, 2] * scale,
        dev[2, 0] * scale,
        dev[2, 1] * scale,
        dev[2, 2] * scale,
    )

    pressure = (trial[0, 0] + trial[1, 1] + trial[2, 2]) / 3.0
    pressure -= dlambda * alpha * (lame + 2.0 * shear / 3.0)

    corrected[0, 0] += pressure
    corrected[1, 1] += pressure
    corrected[2, 2] += pressure

    new_state = MaterialState()
    new_state.equivalent_plastic_strain = state.equivalent_plastic_strain + math.sqrt(2.0 / 3.0) * dlambda
    new_state.hardening_variable = state.hardening_variable + dlambda

    plastic_work = dlambda * (f + mat.yield_strength)
    return corrected, new_state, plastic_work


@dataclass
class MohrCoulomb:
    youngs_modulus: float
    poisson_ratio: float
    cohesion: float
    friction_angle: float
    dilation_angle: float
    tensile_limit: float = 0.0
    hardening_modulus: float = 0.0

    def as_struct(self) -> "MohrCoulombStruct":
        return MohrCoulombStruct(
            elastic=LinearElasticParams(
                youngs_modulus=self.youngs_modulus,
                poisson_ratio=self.poisson_ratio,
            ),
            cohesion=self.cohesion,
            friction_angle=self.friction_angle,
            dilation_angle=self.dilation_angle,
            tensile_limit=self.tensile_limit,
            hardening=self.hardening_modulus,
        )


@wp.struct
class MohrCoulombStruct:
    elastic: LinearElasticParams
    cohesion: float
    friction_angle: float
    dilation_angle: float
    tensile_limit: float
    hardening: float


@wp.func
def mohr_coulomb_yield(
    stress: wp.mat33f,
    state: MaterialState,
    mat: MohrCoulombStruct,
) -> float:
    dev = deviatoric_part(stress)
    j2 = second_invariant(dev)
    pressure = (stress[0, 0] + stress[1, 1] + stress[2, 2]) / 3.0
    sin_phi = wp.sin(mat.friction_angle)
    cos_phi = wp.cos(mat.friction_angle)
    c = mat.cohesion + mat.hardening * state.equivalent_plastic_strain
    return wp.sqrt(j2) + pressure * sin_phi - c * cos_phi


@wp.func
def radial_return_mohr_coulomb(
    trial: wp.mat33f,
    state: MaterialState,
    mat: MohrCoulombStruct,
) -> Tuple[wp.mat33f, MaterialState, float]:
    f = mohr_coulomb_yield(trial, state, mat)
    if f <= 0.0:
        return trial, state, 0.0

    dev = deviatoric_part(trial)
    norm_dev = wp.sqrt(2.0 * second_invariant(dev) + 1.0e-16)
    sin_psi = wp.sin(mat.dilation_angle)
    lame, shear = _lame_shear(mat.elastic)
    denom = 2.0 * shear + (lame + 2.0 * shear / 3.0) * sin_psi
    dlambda = f / denom
    scale = wp.max(0.0, 1.0 - dlambda * 2.0 * shear / norm_dev)

    corrected = wp.mat33f(
        dev[0, 0] * scale,
        dev[0, 1] * scale,
        dev[0, 2] * scale,
        dev[1, 0] * scale,
        dev[1, 1] * scale,
        dev[1, 2] * scale,
        dev[2, 0] * scale,
        dev[2, 1] * scale,
        dev[2, 2] * scale,
    )

    pressure = (trial[0, 0] + trial[1, 1] + trial[2, 2]) / 3.0
    pressure -= dlambda * sin_psi * (lame + 2.0 * shear / 3.0)

    pressure = wp.max(-mat.tensile_limit, pressure)

    corrected[0, 0] += pressure
    corrected[1, 1] += pressure
    corrected[2, 2] += pressure

    new_state = MaterialState()
    new_state.equivalent_plastic_strain = state.equivalent_plastic_strain + math.sqrt(2.0 / 3.0) * dlambda
    new_state.hardening_variable = state.hardening_variable + dlambda

    plastic_work = dlambda * (f + mat.cohesion)
    return corrected, new_state, plastic_work


__all__ = [
    "MaterialState",
    "LinearElasticParams",
    "elastic_trial_stress",
    "DruckerPrager",
    "DruckerPragerStruct",
    "radial_return_drucker_prager",
    "MohrCoulomb",
    "MohrCoulombStruct",
    "radial_return_mohr_coulomb",
]

