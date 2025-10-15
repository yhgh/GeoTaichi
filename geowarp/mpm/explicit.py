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
    origin: wp.vec3f,
    inv_dx: float,
    gx: int,
    gy: int,
    gz: int,
) -> LinearWeights:
    """Return trilinear weights for the eight surrounding grid nodes."""

    cell = (xp - origin) * inv_dx

    bx = int(wp.floor(cell.x))
    by = int(wp.floor(cell.y))
    bz = int(wp.floor(cell.z))

    bx = wp.max(-1, wp.min(bx, gx - 1))
    by = wp.max(-1, wp.min(by, gy - 1))
    bz = wp.max(-1, wp.min(bz, gz - 1))

    base_vec = wp.vec3f(float(bx), float(by), float(bz))
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

    ix0 = bx
    ix1 = bx + 1
    iy0 = by
    iy1 = by + 1
    iz0 = bz
    iz1 = bz + 1

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


@wp.func
def _clamp_position(
    pos: wp.vec3f,
    lower: wp.vec3f,
    upper: wp.vec3f,
) -> wp.vec3f:
    eps = 1.0e-6
    return wp.vec3f(
        wp.max(lower.x + eps, wp.min(upper.x - eps, pos.x)),
        wp.max(lower.y + eps, wp.min(upper.y - eps, pos.y)),
        wp.max(lower.z + eps, wp.min(upper.z - eps, pos.z)),
    )


@wp.kernel
def clear_grid(
    grid_m: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    grid_m[gid] = 0.0
    grid_v[gid] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def p2g(
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    m: wp.array(dtype=float),
    grid_m: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3f),
    origin: wp.vec3f,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i = wp.tid()
    xp = x[i]
    weights = weight_linear(xp, origin, inv_dx, nx, ny, nz)

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
        wp.atomic_add(grid_v, node, v[i] * contrib_m)


@wp.kernel
def normalize_grid(
    grid_m: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    mass = grid_m[gid]
    if mass > 0.0:
        grid_v[gid] = grid_v[gid] / mass
    else:
        grid_v[gid] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def copy_grid(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    dst[gid] = src[gid]


@wp.kernel
def grid_op(
    grid_m: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3f),
    gravity: wp.vec3f,
    dt: float,
    damping: float,
    nx: int,
    ny: int,
    nz: int,
):
    gid = wp.tid()
    if grid_m[gid] <= 0.0:
        grid_v[gid] = wp.vec3f(0.0, 0.0, 0.0)
        return

    vel = grid_v[gid] + gravity * dt
    vel = vel * (1.0 - damping)

    nx_nodes = nx + 1
    ny_nodes = ny + 1
    ix = gid % nx_nodes
    iy = (gid // nx_nodes) % ny_nodes
    iz = gid // (nx_nodes * ny_nodes)

    if ix == 0 and vel.x < 0.0:
        vel.x = 0.0
    if ix == nx and vel.x > 0.0:
        vel.x = 0.0
    if iy == 0 and vel.y < 0.0:
        vel.y = 0.0
    if iy == ny and vel.y > 0.0:
        vel.y = 0.0
    if iz == 0 and vel.z < 0.0:
        vel.z = 0.0
    if iz == nz and vel.z > 0.0:
        vel.z = 0.0

    grid_v[gid] = vel


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    grid_m: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3f),
    grid_v_prev: wp.array(dtype=wp.vec3f),
    origin: wp.vec3f,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    flip_alpha: float,
    domain_min: wp.vec3f,
    domain_max: wp.vec3f,
):
    i = wp.tid()
    xp = x[i]
    weights = weight_linear(xp, origin, inv_dx, nx, ny, nz)

    v_pic = wp.vec3f(0.0, 0.0, 0.0)
    v_flip = wp.vec3f(0.0, 0.0, 0.0)

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

        if grid_m[node] > 0.0:
            node_v = grid_v[node]
            prev_v = grid_v_prev[node]
            v_pic = v_pic + node_v * w
            v_flip = v_flip + (node_v - prev_v) * w

    new_v = v_pic * (1.0 - flip_alpha) + (v[i] + v_flip) * flip_alpha
    new_x = _clamp_position(xp + new_v * dt, domain_min, domain_max)

    v[i] = new_v
    x[i] = new_x


@wp.kernel
def apply_external_forces(
    v: wp.array(dtype=wp.vec3f),
    m: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3f),
    dt: float,
):
    i = wp.tid()
    mass = wp.max(m[i], 1.0e-8)
    v[i] = v[i] + f[i] * (dt / mass)


@dataclass
class ExplicitMPMSolver:
    """Minimal explicit MPM integrator using linear shape functions."""

    positions: Sequence[Sequence[float]]
    velocities: Optional[Sequence[Sequence[float]]] = None
    masses: Optional[Sequence[float]] = None
    dx: float = 0.1
    grid_dims: Tuple[int, int, int] = (16, 16, 16)
    grid_origin: Sequence[float] = (0.0, 0.0, 0.0)
    device: Optional[str] = None
    gravity: Sequence[float] = (0.0, -9.81, 0.0)
    flip_alpha: float = 0.0
    grid_damping: float = 0.0

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
        self.grid_v = wp.zeros(nodes, dtype=wp.vec3f, device=dev)
        self.grid_v_prev = wp.zeros(nodes, dtype=wp.vec3f, device=dev)

        self.device = dev
        self.dx = float(self.dx)
        self.inv_dx = 1.0 / self.dx
        self.origin = wp.vec3f(*self.grid_origin)
        self.domain_min = self.origin
        extent = wp.vec3f(float(nx) * self.dx, float(ny) * self.dx, float(nz) * self.dx)
        self.domain_max = wp.vec3f(
            self.origin.x + extent.x,
            self.origin.y + extent.y,
            self.origin.z + extent.z,
        )
        self.gravity_vec = wp.vec3f(*self.gravity)

    def set_gravity(self, gravity: Sequence[float]) -> None:
        self.gravity_vec = wp.vec3f(*gravity)

    def step(
        self,
        dt: float,
        *,
        gravity: Optional[Sequence[float]] = None,
        flip_alpha: Optional[float] = None,
        external_forces: Optional[wp.array] = None,
    ) -> None:
        nx, ny, nz = self.grid_dims
        grav_vec = self.gravity_vec if gravity is None else wp.vec3f(*gravity)
        flip = self.flip_alpha if flip_alpha is None else float(flip_alpha)

        nodes = self.grid_m.shape[0]

        if external_forces is not None:
            wp.launch(
                apply_external_forces,
                dim=self.x.shape[0],
                inputs=[self.v, self.m, external_forces, dt],
                device=self.device,
            )

        wp.launch(
            clear_grid,
            dim=nodes,
            inputs=[self.grid_m, self.grid_v],
            device=self.device,
        )

        wp.launch(
            p2g,
            dim=self.x.shape[0],
            inputs=[
                self.x,
                self.v,
                self.m,
                self.grid_m,
                self.grid_v,
                self.origin,
                self.inv_dx,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )

        wp.launch(
            normalize_grid,
            dim=nodes,
            inputs=[self.grid_m, self.grid_v],
            device=self.device,
        )

        wp.launch(
            copy_grid,
            dim=nodes,
            inputs=[self.grid_v, self.grid_v_prev],
            device=self.device,
        )

        wp.launch(
            grid_op,
            dim=nodes,
            inputs=[
                self.grid_m,
                self.grid_v,
                grav_vec,
                dt,
                self.grid_damping,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )

        wp.launch(
            g2p,
            dim=self.x.shape[0],
            inputs=[
                self.x,
                self.v,
                self.grid_m,
                self.grid_v,
                self.grid_v_prev,
                self.origin,
                self.inv_dx,
                nx,
                ny,
                nz,
                dt,
                flip,
                self.domain_min,
                self.domain_max,
            ],
            device=self.device,
        )
