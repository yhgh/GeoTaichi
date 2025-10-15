"""Boundary interaction helpers for Warp-based DEM simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, TYPE_CHECKING

import warp as wp

from ..geom.mesh import TriangleMesh
from ..geom.volume import SignedDistanceVolume

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .core import DEMSystem


@wp.kernel
def _mesh_contact_forces(
    mesh_id: wp.uint64,
    bvh_id: wp.uint64,
    vertices: wp.array(dtype=wp.vec3f),
    indices: wp.array(dtype=wp.vec3i),
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3f),
    stiffness: float,
    damping: float,
    friction: float,
    thickness: float,
    boundary_linear_vel: wp.vec3f,
    boundary_angular_vel: wp.vec3f,
    boundary_origin: wp.vec3f,
):
    i = wp.tid()

    xi = x[i]
    vi = v[i]
    radius = r[i]
    max_dist = radius + thickness

    hit, face, u, w, sign = wp.mesh_query_point(mesh_id, bvh_id, xi, max_dist)

    if hit:
        tri = indices[face]
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        bary_w = 1.0 - u - w
        closest = v0 * bary_w + v1 * u + v2 * w
        normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
        dist = wp.dot(xi - closest, normal)
        penetration = max_dist - dist
        if penetration > 0.0:
            boundary_vel = boundary_linear_vel + wp.cross(boundary_angular_vel, closest - boundary_origin)
            rel_v = vi - boundary_vel
            normal_speed = wp.dot(rel_v, normal)
            fn = stiffness * penetration - damping * normal_speed
            normal_force = fn * normal
            tangential = rel_v - normal_speed * normal
            tangent_len = wp.length(tangential)
            friction_force = wp.vec3f(0.0, 0.0, 0.0)
            if friction > 0.0 and tangent_len > 1.0e-6:
                limit = friction * wp.length(normal_force)
                tangential_dir = tangential / (tangent_len + 1.0e-6)
                friction_force = -limit * tangential_dir
            total = normal_force + friction_force
            wp.atomic_add(force, i, total)


@wp.kernel
def _volume_contact_forces(
    volume_id: wp.uint64,
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3f),
    stiffness: float,
    damping: float,
):
    i = wp.tid()

    xi = x[i]
    vi = v[i]
    sdf = wp.volume_sample_world(volume_id, xi)
    penetration = r[i] - sdf
    if penetration > 0.0:
        grad = wp.volume_gradient_world(volume_id, xi)
        normal = wp.normalize(grad)
        normal_speed = wp.dot(vi, normal)
        fn = stiffness * penetration - damping * normal_speed
        wp.atomic_add(force, i, fn * normal)


@wp.kernel
def _wrap_periodic_positions(
    x: wp.array(dtype=wp.vec3f),
    lower: wp.vec3f,
    upper: wp.vec3f,
    extent: wp.vec3f,
):
    i = wp.tid()
    xi = x[i]

    if extent[0] > 0.0:
        while xi[0] < lower[0]:
            xi[0] += extent[0]
        while xi[0] >= upper[0]:
            xi[0] -= extent[0]
    if extent[1] > 0.0:
        while xi[1] < lower[1]:
            xi[1] += extent[1]
        while xi[1] >= upper[1]:
            xi[1] -= extent[1]
    if extent[2] > 0.0:
        while xi[2] < lower[2]:
            xi[2] += extent[2]
        while xi[2] >= upper[2]:
            xi[2] -= extent[2]

    x[i] = xi


@wp.kernel
def dem_contacts_periodic(
    grid_id: wp.uint64,
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3f),
    radius: float,
    k_n: float,
    c_n: float,
    offsets: wp.array(dtype=wp.vec3f),
    num_offsets: int,
):
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

    for oi in range(num_offsets):
        offset = offsets[oi]
        query_offset = wp.hash_grid_query(grid_id, xi + offset, radius)
        j = int(0)
        while wp.hash_grid_query_next(query_offset, j):
            if j == i:
                continue
            xj = x[j] - offset
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
class MeshBoundary:
    """Triangle-mesh collider used for DEM boundary conditions."""

    mesh: TriangleMesh
    stiffness: float = 1.0e5
    damping: float = 1.0e2
    friction: float = 0.0
    thickness: float = 0.0
    linear_velocity: Sequence[float] = (0.0, 0.0, 0.0)
    angular_velocity: Sequence[float] = (0.0, 0.0, 0.0)
    origin: Sequence[float] = (0.0, 0.0, 0.0)

    def apply(self, system: "DEMSystem") -> None:
        """Accumulate contact forces onto the DEM system."""

        mesh = self.mesh
        if mesh.bvh is None:
            mesh.ensure_bvh()
        wp.launch(
            _mesh_contact_forces,
            dim=system.x.shape[0],
            inputs=[
                mesh.mesh_id,
                mesh.bvh_id,
                mesh.vertex_array,
                mesh.index_array,
                system.x,
                system.v,
                system.r,
                system.force,
                self.stiffness,
                self.damping,
                self.friction,
                self.thickness,
                wp.vec3f(*self.linear_velocity),
                wp.vec3f(*self.angular_velocity),
                wp.vec3f(*self.origin),
            ],
            device=system.device,
        )


@dataclass
class VolumeBoundary:
    """Level-set collider sampled from a signed-distance volume."""

    volume: SignedDistanceVolume
    stiffness: float = 1.0e5
    damping: float = 1.0e2

    def apply(self, system: "DEMSystem") -> None:
        wp.launch(
            _volume_contact_forces,
            dim=system.x.shape[0],
            inputs=[
                self.volume.volume_id,
                system.x,
                system.v,
                system.r,
                system.force,
                self.stiffness,
                self.damping,
            ],
            device=system.device,
        )


@dataclass
class PeriodicBoundary:
    """Periodic domain helper that wraps particle positions and offsets queries."""

    lower: Sequence[float]
    upper: Sequence[float]
    periodic_axes: Sequence[bool] = (True, True, True)
    device: Optional[str] = None
    _offsets: Optional[wp.array] = field(init=False, default=None)

    def __post_init__(self) -> None:
        dev = self.device or wp.get_device()
        self.device = dev
        lx, ly, lz = self.lower
        ux, uy, uz = self.upper
        ex = ux - lx
        ey = uy - ly
        ez = uz - lz
        lower_vec = wp.vec3f(lx, ly, lz)
        upper_vec = wp.vec3f(ux, uy, uz)
        extent_vec = wp.vec3f(ex, ey, ez)
        self.lower_vec = lower_vec
        self.upper_vec = upper_vec
        self.extent_vec = extent_vec
        offsets = []
        x_offsets = [-ex, 0.0, ex] if self.periodic_axes[0] and ex > 0.0 else [0.0]
        y_offsets = [-ey, 0.0, ey] if self.periodic_axes[1] and ey > 0.0 else [0.0]
        z_offsets = [-ez, 0.0, ez] if self.periodic_axes[2] and ez > 0.0 else [0.0]
        for dx in x_offsets:
            for dy in y_offsets:
                for dz in z_offsets:
                    if abs(dx) < 1.0e-6 and abs(dy) < 1.0e-6 and abs(dz) < 1.0e-6:
                        continue
                    offsets.append((dx, dy, dz))
        if offsets:
            self._offsets = wp.array(offsets, dtype=wp.vec3f, device=dev)
        else:
            self._offsets = None

    @property
    def offsets(self) -> Optional[wp.array]:
        return self._offsets

    def offset_count(self) -> int:
        return 0 if self._offsets is None else self._offsets.shape[0]

    def wrap(self, positions: wp.array) -> None:
        wp.launch(
            _wrap_periodic_positions,
            dim=positions.shape[0],
            inputs=[positions, self.lower_vec, self.upper_vec, self.extent_vec],
            device=self.device,
        )
