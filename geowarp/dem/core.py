"""Minimal discrete element method (DEM) kernels backed by Warp hash grids.

This module implements a light-weight particle contact model intended to
mirror GeoTaichi's DEM stepping behaviour. It focuses on particle-particle
contacts via a linear normal spring with viscous damping while relying on
Warp's `HashGrid` acceleration structure to find neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import warp as wp

from .boundary import PeriodicBoundary, dem_contacts_periodic


@wp.kernel
def _zero_vec3(arr: wp.array(dtype=wp.vec3f)):
    """Set a vector array to zero."""

    i = wp.tid()
    arr[i] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def _integrate_semi_implicit(
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    force: wp.array(dtype=wp.vec3f),
    mass: wp.array(dtype=float),
    gravity: wp.vec3f,
    dt: float,
    damping: float,
):
    """Semi-implicit Euler integration for DEM particles."""

    i = wp.tid()
    inv_m = 1.0 / wp.max(mass[i], 1e-8)
    acc = gravity + force[i] * inv_m
    v_new = v[i] + acc * dt
    v[i] = v_new * (1.0 - damping)
    x[i] = x[i] + v[i] * dt
    force[i] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def dem_contacts(
    grid_id: wp.uint64,
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3f),
    radius: float,
    k_n: float,
    c_n: float,
):
    """Accumulate pairwise DEM forces using a linear dashpot law."""

    i = wp.tid()
    xi = x[i]
    vi = v[i]
    ri = r[i]

    query = wp.hash_grid_query(grid_id, xi, radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue

        xj = x[j]
        rj = r[j]
        d = xj - xi
        dist = wp.length(d)
        overlap = (ri + rj) - dist

        if overlap > 0.0:
            n = d / (dist + 1.0e-7)
            rel_v = vi - v[j]
            fn = k_n * overlap * n - c_n * wp.dot(rel_v, n) * n
            wp.atomic_add(force, i, -fn)
            wp.atomic_add(force, j, fn)


@dataclass
class DEMSystem:
    """Simple DEM particle container built on Warp primitives."""

    positions: Sequence[Sequence[float]]
    radii: Sequence[float]
    masses: Optional[Sequence[float]] = None
    device: Optional[str] = None
    gravity: Sequence[float] = (0.0, -9.81, 0.0)
    damping: float = 0.0
    periodic: Optional[PeriodicBoundary] = None
    boundaries: Sequence[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        dev = self.device or wp.get_device()
        pos = self.positions
        if self.masses is None:
            mass_values = [1.0] * len(pos)
        else:
            mass_values = self.masses

        self.x = wp.array(pos, dtype=wp.vec3f, device=dev)
        self.v = wp.zeros(self.x.shape, dtype=wp.vec3f, device=dev)
        self.r = wp.array(self.radii, dtype=float, device=dev)
        self.m = wp.array(mass_values, dtype=float, device=dev)
        self.force = wp.zeros(self.x.shape, dtype=wp.vec3f, device=dev)
        self.grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=dev)
        self._gravity_vec = wp.vec3f(*self.gravity)
        self.device = dev
        self._periodic = self.periodic
        self._boundaries: List[object] = list(self.boundaries)

    def set_gravity(self, gravity: Sequence[float]) -> None:
        """Update the global gravity vector."""

        self._gravity_vec = wp.vec3f(*gravity)

    def set_periodic_boundary(self, periodic: Optional[PeriodicBoundary]) -> None:
        """Attach or clear the periodic boundary handler."""

        self._periodic = periodic

    def add_boundary(self, boundary: object) -> None:
        """Register a mesh or volume boundary condition."""

        self._boundaries.append(boundary)

    def step(
        self,
        dt: float,
        search_radius: float,
        k_n: float,
        c_n: float,
        *,
        contact_callback: Optional[Callable[["DEMSystem"], None]] = None,
    ) -> None:
        """Advance the DEM system by one timestep."""

        # Reset force accumulators before evaluating contacts.
        wp.launch(_zero_vec3, dim=self.force.shape[0], inputs=[self.force], device=self.device)

        # Rebuild the spatial hash grid with the current particle state.
        self.grid.build(points=self.x, radius=search_radius)

        # Compute pairwise forces with optional periodic tiling.
        periodic = self._periodic
        if periodic is None or periodic.offset_count() == 0:
            wp.launch(
                dem_contacts,
                dim=self.x.shape[0],
                inputs=[
                    self.grid.id,
                    self.x,
                    self.v,
                    self.r,
                    self.force,
                    search_radius,
                    k_n,
                    c_n,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                dem_contacts_periodic,
                dim=self.x.shape[0],
                inputs=[
                    self.grid.id,
                    self.x,
                    self.v,
                    self.r,
                    self.force,
                    search_radius,
                    k_n,
                    c_n,
                    periodic.offsets,
                    periodic.offset_count(),
                ],
                device=self.device,
            )

        # Apply additional boundary contacts.
        for boundary in self._boundaries:
            if hasattr(boundary, "apply"):
                boundary.apply(self)

        if contact_callback is not None:
            contact_callback(self)

        # Integrate velocities and positions.
        wp.launch(
            _integrate_semi_implicit,
            dim=self.x.shape[0],
            inputs=[
                self.x,
                self.v,
                self.force,
                self.m,
                self._gravity_vec,
                dt,
                self.damping,
            ],
            device=self.device,
        )

        if periodic is not None:
            periodic.wrap(self.x)
