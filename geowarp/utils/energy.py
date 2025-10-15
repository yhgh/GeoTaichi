"""Energy accounting helpers shared between MPM and DEM solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import warp as wp


def _as_wp_array(value, dtype, device):
    if isinstance(value, wp.array):
        return value
    return wp.array(value, dtype=dtype, device=device)


@wp.kernel
def _kinetic_energy_kernel(
    masses: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = 0.5 * masses[i] * wp.dot(velocities[i], velocities[i])


@wp.kernel
def _potential_energy_kernel(
    masses: wp.array(dtype=float),
    positions: wp.array(dtype=wp.vec3f),
    gravity: wp.vec3f,
    reference: float,
    out: wp.array(dtype=float),
):
    i = wp.tid()
    relative = positions[i] - wp.vec3f(0.0, reference, 0.0)
    out[i] = -masses[i] * wp.dot(gravity, relative)


@wp.kernel
def _work_kernel(
    contributions: wp.array(dtype=float),
    out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = contributions[i]


def compute_kinetic_energy(
    masses,
    velocities,
    device: Optional[str] = None,
) -> float:
    """Return total kinetic energy for the supplied particle arrays."""

    if isinstance(velocities, wp.array):
        device = velocities.device
    elif device is None:
        device = wp.get_device().alias

    masses_wp = _as_wp_array(masses, dtype=float, device=device)
    velocities_wp = _as_wp_array(velocities, dtype=wp.vec3f, device=device)
    tmp = wp.empty(masses_wp.shape[0], dtype=float, device=device)

    wp.launch(
        _kinetic_energy_kernel,
        dim=masses_wp.shape[0],
        inputs=[masses_wp, velocities_wp, tmp],
        device=device,
    )
    return float(wp.to_numpy(tmp).sum())


def compute_potential_energy(
    masses,
    positions,
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
    reference_height: float = 0.0,
    device: Optional[str] = None,
) -> float:
    """Return gravitational potential energy relative to ``reference_height``."""

    if isinstance(positions, wp.array):
        device = positions.device
    elif device is None:
        device = wp.get_device().alias

    masses_wp = _as_wp_array(masses, dtype=float, device=device)
    positions_wp = _as_wp_array(positions, dtype=wp.vec3f, device=device)
    tmp = wp.empty(masses_wp.shape[0], dtype=float, device=device)

    wp.launch(
        _potential_energy_kernel,
        dim=masses_wp.shape[0],
        inputs=[masses_wp, positions_wp, wp.vec3f(*gravity), reference_height, tmp],
        device=device,
    )
    return float(wp.to_numpy(tmp).sum())


def compute_contact_energy(
    normal_work,
    friction_work=None,
    damping_work=None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Return accumulated contact, friction, and damping work terms."""

    if isinstance(normal_work, wp.array):
        device = normal_work.device
    elif device is None:
        device = wp.get_device().alias

    normal_wp = _as_wp_array(normal_work, dtype=float, device=device)
    tmp = wp.empty(normal_wp.shape[0], dtype=float, device=device)
    wp.launch(_work_kernel, dim=normal_wp.shape[0], inputs=[normal_wp, tmp], device=device)
    normal_total = float(wp.to_numpy(tmp).sum())

    friction_total = 0.0
    if friction_work is not None:
        friction_wp = _as_wp_array(friction_work, dtype=float, device=device)
        tmp = wp.empty(friction_wp.shape[0], dtype=float, device=device)
        wp.launch(_work_kernel, dim=friction_wp.shape[0], inputs=[friction_wp, tmp], device=device)
        friction_total = float(wp.to_numpy(tmp).sum())

    damping_total = 0.0
    if damping_work is not None:
        damping_wp = _as_wp_array(damping_work, dtype=float, device=device)
        tmp = wp.empty(damping_wp.shape[0], dtype=float, device=device)
        wp.launch(_work_kernel, dim=damping_wp.shape[0], inputs=[damping_wp, tmp], device=device)
        damping_total = float(wp.to_numpy(tmp).sum())

    return {
        "normal": normal_total,
        "friction": friction_total,
        "damping": damping_total,
        "total": normal_total + friction_total + damping_total,
    }


@dataclass
class EnergyAccumulator:
    """Helper to track energy budgets across simulation subsystems."""

    kinetic: float = 0.0
    potential: float = 0.0
    elastic: float = 0.0
    plastic: float = 0.0
    frictional: float = 0.0
    damping: float = 0.0
    contact: float = 0.0

    def reset(self) -> None:
        self.kinetic = 0.0
        self.potential = 0.0
        self.elastic = 0.0
        self.plastic = 0.0
        self.frictional = 0.0
        self.damping = 0.0
        self.contact = 0.0

    def accumulate_kinetic(self, value: float) -> None:
        self.kinetic += value

    def accumulate_potential(self, value: float) -> None:
        self.potential += value

    def accumulate_elastic(self, value: float) -> None:
        self.elastic += value

    def accumulate_plastic(self, value: float) -> None:
        self.plastic += value

    def accumulate_contact(self, normal: float, friction: float = 0.0, damping: float = 0.0) -> None:
        self.contact += normal
        self.frictional += friction
        self.damping += damping

    def snapshot(self) -> Dict[str, float]:
        return {
            "kinetic": self.kinetic,
            "potential": self.potential,
            "elastic": self.elastic,
            "plastic": self.plastic,
            "frictional": self.frictional,
            "damping": self.damping,
            "contact": self.contact,
        }


__all__ = [
    "EnergyAccumulator",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "compute_contact_energy",
]

