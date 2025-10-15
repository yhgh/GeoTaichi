"""Explicit material point method (MPM) stepping kernels using Warp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import warp as wp


@wp.struct
class LinearWeights:
    nodes0: wp.vec4i
    nodes1: wp.vec4i
    weights0: wp.vec4f
    weights1: wp.vec4f


@wp.func
def _node_index(ix: int, iy: int, iz: int, nx: int, ny: int, nz: int) -> int:
    nx_nodes = nx + 1
    ny_nodes = ny + 1
    ix_clamped = wp.max(0, wp.min(ix, nx))
    iy_clamped = wp.max(0, wp.min(iy, ny))
    iz_clamped = wp.max(0, wp.min(iz, nz))
    return (
        ix_clamped
        + nx_nodes * iy_clamped
        + nx_nodes * ny_nodes * iz_clamped
    )


@wp.func
def weight_linear(
    xp: wp.vec3f,
    base: wp.vec3i,
    dx: float,
    gx: int,
    gy: int,
    gz: int,
) -> LinearWeights:
    """Return trilinear weights for the eight surrounding grid nodes."""

    cell = xp / dx
    base_vec = wp.vec3f(float(base.x), float(base.y), float(base.z))
    fx = cell - base_vec
    fx_x = wp.max(0.0, wp.min(fx.x, 1.0))
    fx_y = wp.max(0.0, wp.min(fx.y, 1.0))
    fx_z = wp.max(0.0, wp.min(fx.z, 1.0))

    wx0 = 1.0 - fx_x
    wx1 = fx_x
    wy0 = 1.0 - fx_y
    wy1 = fx_y
    wz0 = 1.0 - fx_z
    wz1 = fx_z

    ix0 = base.x
    ix1 = base.x + 1
    iy0 = base.y
    iy1 = base.y + 1
    iz0 = base.z
    iz1 = base.z + 1

    weights = LinearWeights()
    weights.nodes0 = wp.vec4i(
        _node_index(ix0, iy0, iz0, gx, gy, gz),
        _node_index(ix1, iy0, iz0, gx, gy, gz),
        _node_index(ix0, iy1, iz0, gx, gy, gz),
        _node_index(ix1, iy1, iz0, gx, gy, gz),
    )
    weights.nodes1 = wp.vec4i(
        _node_index(ix0, iy0, iz1, gx, gy, gz),
        _node_index(ix1, iy0, iz1, gx, gy, gz),
        _node_index(ix0, iy1, iz1, gx, gy, gz),
        _node_index(ix1, iy1, iz1, gx, gy, gz),
    )
    weights.weights0 = wp.vec4f(
        wx0 * wy0 * wz0,
        wx1 * wy0 * wz0,
        wx0 * wy1 * wz0,
        wx1 * wy1 * wz0,
    )
    weights.weights1 = wp.vec4f(
        wx0 * wy0 * wz1,
        wx1 * wy0 * wz1,
        wx0 * wy1 * wz1,
        wx1 * wy1 * wz1,
    )
    return weights


@wp.kernel
def clear_grid(
    grid_m: wp.array(dtype=float),
    grid_p: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    grid_m[gid] = 0.0
    grid_p[gid] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def p2g(
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    m: wp.array(dtype=float),
    grid_m: wp.array(dtype=float),
    grid_p: wp.array(dtype=wp.vec3f),
    dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i = wp.tid()
    xp = x[i]
    base = wp.vec3i(
        int(wp.floor(xp.x / dx)),
        int(wp.floor(xp.y / dx)),
        int(wp.floor(xp.z / dx)),
    )
    weights = weight_linear(xp, base, dx, nx, ny, nz)

    for k in wp.static_range(8):
        if k == 0:
            node = weights.nodes0.x
            w = weights.weights0.x
        elif k == 1:
            node = weights.nodes0.y
            w = weights.weights0.y
        elif k == 2:
            node = weights.nodes0.z
            w = weights.weights0.z
        elif k == 3:
            node = weights.nodes0.w
            w = weights.weights0.w
        elif k == 4:
            node = weights.nodes1.x
            w = weights.weights1.x
        elif k == 5:
            node = weights.nodes1.y
            w = weights.weights1.y
        elif k == 6:
            node = weights.nodes1.z
            w = weights.weights1.z
        else:
            node = weights.nodes1.w
            w = weights.weights1.w

        contrib_m = m[i] * w
        wp.atomic_add(grid_m, node, contrib_m)
        wp.atomic_add(grid_p, node, v[i] * contrib_m)


@wp.kernel
def grid_op(
    grid_m: wp.array(dtype=float),
    grid_p: wp.array(dtype=wp.vec3f),
    gravity: wp.vec3f,
    dt: float,
):
    gid = wp.tid()
    mass = grid_m[gid]
    if mass > 0.0:
        momentum = grid_p[gid]
        vel = momentum / mass
        vel = vel + gravity * dt
        grid_p[gid] = vel * mass


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    grid_m: wp.array(dtype=float),
    grid_p: wp.array(dtype=wp.vec3f),
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
):
    i = wp.tid()
    xp = x[i]
    base = wp.vec3i(
        int(wp.floor(xp.x / dx)),
        int(wp.floor(xp.y / dx)),
        int(wp.floor(xp.z / dx)),
    )
    weights = weight_linear(xp, base, dx, nx, ny, nz)

    v_pic = wp.vec3f(0.0, 0.0, 0.0)
    for k in wp.static_range(8):
        if k == 0:
            node = weights.nodes0.x
            w = weights.weights0.x
        elif k == 1:
            node = weights.nodes0.y
            w = weights.weights0.y
        elif k == 2:
            node = weights.nodes0.z
            w = weights.weights0.z
        elif k == 3:
            node = weights.nodes0.w
            w = weights.weights0.w
        elif k == 4:
            node = weights.nodes1.x
            w = weights.weights1.x
        elif k == 5:
            node = weights.nodes1.y
            w = weights.weights1.y
        elif k == 6:
            node = weights.nodes1.z
            w = weights.weights1.z
        else:
            node = weights.nodes1.w
            w = weights.weights1.w

        mass = grid_m[node]
        if mass > 0.0:
            node_v = grid_p[node] / mass
            v_pic = v_pic + node_v * w

    v[i] = v_pic
    x[i] = xp + v_pic * dt


@dataclass
class ExplicitMPMSolver:
    """Minimal explicit MPM integrator using linear shape functions."""

    positions: Sequence[Sequence[float]]
    velocities: Optional[Sequence[Sequence[float]]] = None
    masses: Optional[Sequence[float]] = None
    dx: float = 0.1
    grid_dims: Tuple[int, int, int] = (16, 16, 16)
    device: Optional[str] = None
    gravity: Sequence[float] = (0.0, -9.81, 0.0)

    def __post_init__(self) -> None:
        dev = self.device or wp.get_device()
        n = len(self.positions)
        if self.velocities is None:
            vel_values = [(0.0, 0.0, 0.0)] * n
        else:
            vel_values = self.velocities
        if self.masses is None:
            mass_values = [1.0] * n
        else:
            mass_values = self.masses

        self.x = wp.array(self.positions, dtype=wp.vec3f, device=dev)
        self.v = wp.array(vel_values, dtype=wp.vec3f, device=dev)
        self.m = wp.array(mass_values, dtype=float, device=dev)

        nx, ny, nz = self.grid_dims
        nodes = (nx + 1) * (ny + 1) * (nz + 1)
        self.grid_m = wp.zeros(nodes, dtype=float, device=dev)
        self.grid_p = wp.zeros(nodes, dtype=wp.vec3f, device=dev)

        self.device = dev
        self.gravity_vec = wp.vec3f(*self.gravity)

    def set_gravity(self, gravity: Sequence[float]) -> None:
        self.gravity_vec = wp.vec3f(*gravity)

    def step(self, dt: float) -> None:
        nx, ny, nz = self.grid_dims

        # Reset grid buffers.
        wp.launch(
            clear_grid,
            dim=self.grid_m.shape[0],
            inputs=[self.grid_m, self.grid_p],
            device=self.device,
        )

        # Particle-to-grid.
        wp.launch(
            p2g,
            dim=self.x.shape[0],
            inputs=[self.x, self.v, self.m, self.grid_m, self.grid_p, self.dx, nx, ny, nz],
            device=self.device,
        )

        # Grid operations (gravity, boundary conditions placeholder).
        wp.launch(
            grid_op,
            dim=self.grid_m.shape[0],
            inputs=[self.grid_m, self.grid_p, self.gravity_vec, dt],
            device=self.device,
        )

        # Grid-to-particle.
        wp.launch(
            g2p,
            dim=self.x.shape[0],
            inputs=[self.x, self.v, self.grid_m, self.grid_p, self.dx, nx, ny, nz, dt],
            device=self.device,
        )
