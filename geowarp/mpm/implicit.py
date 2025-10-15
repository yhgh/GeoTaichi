"""Implicit MPM stepping utilities backed by Warp sparse solvers."""

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
def assemble_rhs(
    grid_m: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3f),
    gravity: wp.vec3f,
    dt: float,
    rhs: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    mass = grid_m[gid]
    if mass > 0.0:
        rhs[gid] = grid_v[gid] * (mass / dt) + gravity * mass
    else:
        rhs[gid] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def assemble_diagonal_triplets(
    grid_m: wp.array(dtype=float),
    stiffness: float,
    dt: float,
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    vals: wp.array(dtype=wp.mat33f),
):
    gid = wp.tid()
    mass = grid_m[gid]
    rows[gid] = gid
    cols[gid] = gid
    diag = (mass / dt) + stiffness
    vals[gid] = wp.mat33f(
        diag,
        0.0,
        0.0,
        0.0,
        diag,
        0.0,
        0.0,
        0.0,
        diag,
    )


@wp.kernel
def apply_solution(
    solution: wp.array(dtype=wp.vec3f),
    grid_v: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    grid_v[gid] = solution[gid]


@wp.kernel
def copy_field(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    dst[gid] = src[gid]


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3f),
    v: wp.array(dtype=wp.vec3f),
    grid_m: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3f),
    origin: wp.vec3f,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    domain_min: wp.vec3f,
    domain_max: wp.vec3f,
):
    i = wp.tid()
    xp = x[i]
    weights = weight_linear(xp, origin, inv_dx, nx, ny, nz)

    new_v = wp.vec3f(0.0, 0.0, 0.0)

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
            new_v = new_v + grid_v[node] * w

    new_x = _clamp_position(xp + new_v * dt, domain_min, domain_max)
    v[i] = new_v
    x[i] = new_x


def _compute_isotropic_stiffness(youngs: float, poisson: float) -> Tuple[float, float]:
    poisson = max(-0.99, min(poisson, 0.49))
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return float(lam), float(mu)


@dataclass
class ImplicitMPMSolver:
    """Implicit MPM integrator using a diagonalized sparse system."""

    positions: Sequence[Sequence[float]]
    velocities: Optional[Sequence[Sequence[float]]] = None
    masses: Optional[Sequence[float]] = None
    dx: float = 0.1
    grid_dims: Tuple[int, int, int] = (16, 16, 16)
    grid_origin: Sequence[float] = (0.0, 0.0, 0.0)
    device: Optional[str] = None
    gravity: Sequence[float] = (0.0, -9.81, 0.0)
    youngs_modulus: float = 1.0e5
    poisson_ratio: float = 0.3

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
        self.grid_rhs = wp.zeros(nodes, dtype=wp.vec3f, device=dev)
        self.solution = wp.zeros(nodes, dtype=wp.vec3f, device=dev)

        self.rows = wp.zeros(nodes, dtype=int, device=dev)
        self.cols = wp.zeros(nodes, dtype=int, device=dev)
        self.vals = wp.zeros(nodes, dtype=wp.mat33f, device=dev)

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

        lam, mu = _compute_isotropic_stiffness(self.youngs_modulus, self.poisson_ratio)
        self.elastic_lambda = lam
        self.elastic_mu = mu

        if not hasattr(wp, "sparse"):
            raise RuntimeError("Warp sparse module is required for the implicit solver")
        matrix_cls = getattr(wp.sparse, "BsrMatrix", None) or getattr(
            wp.sparse, "BSRMatrix", None
        )
        if matrix_cls is None:
            raise RuntimeError("Warp sparse backend does not expose a BSR matrix type")

        try:
            self.matrix = matrix_cls(
                shape=(nodes, nodes),
                block_size=3,
                dtype=float,
                device=dev,
            )
        except TypeError:
            # Older Warp releases use positional construction.
            self.matrix = matrix_cls(
                (nodes, nodes),
                block_size=3,
                dtype=float,
                device=dev,
            )

    def set_gravity(self, gravity: Sequence[float]) -> None:
        self.gravity_vec = wp.vec3f(*gravity)

    def assemble_system(self, dt: float) -> None:
        nx, ny, nz = self.grid_dims
        nodes = self.grid_m.shape[0]

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
            assemble_rhs,
            dim=nodes,
            inputs=[self.grid_m, self.grid_v, self.gravity_vec, dt, self.grid_rhs],
            device=self.device,
        )

        stiffness = float(self.elastic_lambda + 2.0 * self.elastic_mu)
        wp.launch(
            assemble_diagonal_triplets,
            dim=nodes,
            inputs=[self.grid_m, stiffness, dt, self.rows, self.cols, self.vals],
            device=self.device,
        )

        bsr_from_triplets = getattr(wp.sparse, "bsr_from_triplets", None)
        if bsr_from_triplets is None:
            raise RuntimeError("Warp sparse backend missing bsr_from_triplets helper")
        try:
            bsr_from_triplets(
                self.rows,
                self.cols,
                self.vals,
                self.matrix,
            )
        except TypeError:
            bsr_from_triplets(
                self.matrix,
                self.rows,
                self.cols,
                self.vals,
            )

    def solve_grid(self, tol: float = 1.0e-6, max_iters: int = 200) -> None:
        wp.launch(
            copy_field,
            dim=self.solution.shape[0],
            inputs=[self.grid_v, self.solution],
            device=self.device,
        )

        cg_solver = getattr(wp.sparse, "bsr_cg", None)
        if cg_solver is None:
            raise RuntimeError("Warp sparse backend missing bsr_cg solver")
        cg_solver(
            self.matrix,
            self.grid_rhs,
            self.solution,
            max_iters=max_iters,
            tol=tol,
        )

        wp.launch(
            apply_solution,
            dim=self.grid_v.shape[0],
            inputs=[self.solution, self.grid_v],
            device=self.device,
        )

    def step(
        self,
        dt: float,
        *,
        gravity: Optional[Sequence[float]] = None,
        tol: float = 1.0e-6,
        max_iters: int = 200,
    ) -> None:
        if gravity is not None:
            self.gravity_vec = wp.vec3f(*gravity)

        self.assemble_system(dt)
        self.solve_grid(tol=tol, max_iters=max_iters)

        nx, ny, nz = self.grid_dims
        wp.launch(
            g2p,
            dim=self.x.shape[0],
            inputs=[
                self.x,
                self.v,
                self.grid_m,
                self.grid_v,
                self.origin,
                self.inv_dx,
                nx,
                ny,
                nz,
                dt,
                self.domain_min,
                self.domain_max,
            ],
            device=self.device,
        )
