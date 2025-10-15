"""Coupling utilities bridging DEM and explicit MPM solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import warp as wp

from ..dem.core import DEMSystem
from ..mpm.explicit import ExplicitMPMSolver


@wp.kernel
def _zero_vec3(arr: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    arr[i] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def mpm_dem_contacts(
    mpm_grid_id: wp.uint64,
    dem_x: wp.array(dtype=wp.vec3f),
    dem_v: wp.array(dtype=wp.vec3f),
    dem_r: wp.array(dtype=float),
    dem_force: wp.array(dtype=wp.vec3f),
    mpm_x: wp.array(dtype=wp.vec3f),
    mpm_v: wp.array(dtype=wp.vec3f),
    mpm_force: wp.array(dtype=wp.vec3f),
    mpm_r: wp.array(dtype=float),
    search_radius: float,
    k_n: float,
    c_n: float,
):
    i = wp.tid()
    xi = dem_x[i]
    vi = dem_v[i]
    ri = dem_r[i]

    query = wp.hash_grid_query(mpm_grid_id, xi, search_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        xp = mpm_x[j]
        rp = mpm_r[j]

        d = xp - xi
        dist = wp.length(d)
        overlap = (ri + rp) - dist

        if overlap > 0.0:
            n = d / (dist + 1.0e-7)
            rel_v = vi - mpm_v[j]
            fn = k_n * overlap * n - c_n * wp.dot(rel_v, n) * n
            wp.atomic_add(dem_force, i, -fn)
            wp.atomic_add(mpm_force, j, fn)


@dataclass
class MPDEMCoupling:
    """Bridge explicit MPM and DEM systems with penalty-based contacts."""

    mpm: ExplicitMPMSolver
    dem: DEMSystem
    mpm_radii: Optional[Sequence[float]] = None
    grid_resolution: int = 128

    def __post_init__(self) -> None:
        self.device = self.mpm.device
        if str(self.dem.device) != str(self.device):
            raise ValueError("MPM and DEM systems must live on the same device")

        radius_default = 0.5 * getattr(self.mpm, "dx", 1.0)
        if self.mpm_radii is None:
            radii_values = [radius_default] * self.mpm.x.shape[0]
        else:
            if isinstance(self.mpm_radii, Sequence):
                radii_values = [float(r) for r in self.mpm_radii]
                if len(radii_values) != self.mpm.x.shape[0]:
                    raise ValueError(
                        "MPM radii sequence must match the number of particles"
                    )
            else:
                radii_values = [float(self.mpm_radii)] * self.mpm.x.shape[0]

        self._mpm_r = wp.array(radii_values, dtype=float, device=self.device)
        self._mpm_force = wp.zeros(self.mpm.x.shape, dtype=wp.vec3f, device=self.device)
        self._mpm_grid = wp.HashGrid(
            dim_x=self.grid_resolution,
            dim_y=self.grid_resolution,
            dim_z=self.grid_resolution,
            device=self.device,
        )

    def set_mpm_radii(self, radii: Sequence[float]) -> None:
        """Update the effective radii used for MPM particles."""

        if len(radii) != self._mpm_r.shape[0]:
            raise ValueError("Radius sequence must match the number of MPM particles")
        self._mpm_r = wp.array([float(r) for r in radii], dtype=float, device=self.device)

    def _compute_coupling_forces(
        self,
        dem_system: DEMSystem,
        search_radius: float,
        k_n: float,
        c_n: float,
    ) -> None:
        wp.launch(
            _zero_vec3,
            dim=self._mpm_force.shape[0],
            inputs=[self._mpm_force],
            device=self.device,
        )

        self._mpm_grid.build(points=self.mpm.x, radius=search_radius)

        wp.launch(
            mpm_dem_contacts,
            dim=dem_system.x.shape[0],
            inputs=[
                self._mpm_grid.id,
                dem_system.x,
                dem_system.v,
                dem_system.r,
                dem_system.force,
                self.mpm.x,
                self.mpm.v,
                self._mpm_force,
                self._mpm_r,
                search_radius,
                k_n,
                c_n,
            ],
            device=self.device,
        )

    def step(
        self,
        dt: float,
        *,
        search_radius: float,
        k_n: float,
        c_n: float,
        gravity: Optional[Sequence[float]] = None,
        flip_alpha: Optional[float] = None,
    ) -> None:
        """Advance the coupled system by a single timestep."""

        def _callback(system: DEMSystem) -> None:
            self._compute_coupling_forces(system, search_radius, k_n, c_n)

        self.dem.step(
            dt,
            search_radius,
            k_n,
            c_n,
            contact_callback=_callback,
        )

        self.mpm.step(
            dt,
            gravity=gravity,
            flip_alpha=flip_alpha,
            external_forces=self._mpm_force,
        )

    @property
    def mpm_forces(self) -> wp.array:
        """Return the last computed MPM contact forces."""

        return self._mpm_force

