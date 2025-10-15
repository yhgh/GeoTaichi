"""Execute lightweight GeoWarp regression scenarios and update baselines."""

from __future__ import annotations

import argparse
from typing import Iterable, Sequence

from geowarp.backend import init as backend_init
from geowarp.dem.core import DEMSystem
from geowarp.mpm.explicit import ExplicitMPMSolver
from geowarp.mpdem.coupling import MPDEMCoupling

from .common import format_metrics, record_metrics, timed_step


def _grid_positions(counts: Sequence[int], spacing: Sequence[float], origin: Sequence[float]):
    gx, gy, gz = counts
    sx, sy, sz = spacing
    ox, oy, oz = origin
    points = []
    for ix in range(gx):
        for iy in range(gy):
            for iz in range(gz):
                points.append((ox + ix * sx, oy + iy * sy, oz + iz * sz))
    return points


def run_dem(dt: float = 2.5e-4, steps: int = 20) -> None:
    positions = _grid_positions((6, 4, 1), (0.05, 0.05, 0.05), (0.0, 0.05, 0.0))
    radii = [0.0225] * len(positions)
    masses = [0.8] * len(positions)

    dem = DEMSystem(positions=positions, radii=radii, masses=masses, damping=0.05)

    def _step():
        dem.step(dt, search_radius=0.06, k_n=2.5e4, c_n=15.0)

    step_time = timed_step(_step, steps)
    metrics = record_metrics(
        "dem_column_collapse", dem.x, dem.v, dem.m, (0.0, -9.81, 0.0), step_time
    )
    print("DEM column collapse:", format_metrics(metrics))


def run_mpm(dt: float = 1.0e-3, steps: int = 15) -> None:
    positions = _grid_positions((5, 5, 3), (0.08, 0.08, 0.08), (0.0, 0.12, 0.0))
    masses = [1.0] * len(positions)

    solver = ExplicitMPMSolver(
        positions=positions,
        masses=masses,
        dx=0.08,
        grid_dims=(12, 12, 6),
        grid_origin=(0.0, 0.0, 0.0),
        gravity=(0.0, -9.81, 0.0),
        flip_alpha=0.0,
    )

    def _step():
        solver.step(dt)

    step_time = timed_step(_step, steps)
    metrics = record_metrics(
        "mpm_column_collapse", solver.x, solver.v, solver.m, (0.0, -9.81, 0.0), step_time
    )
    print("MPM column collapse:", format_metrics(metrics))


def run_mpdem(dt: float = 7.5e-4, steps: int = 12) -> None:
    mpm_positions = _grid_positions((4, 4, 2), (0.07, 0.07, 0.07), (0.0, 0.1, 0.0))
    dem_positions = _grid_positions((3, 3, 1), (0.09, 0.09, 0.09), (0.1, 0.05, 0.0))

    mpm = ExplicitMPMSolver(
        positions=mpm_positions,
        dx=0.07,
        grid_dims=(10, 10, 6),
        grid_origin=(0.0, 0.0, 0.0),
        gravity=(0.0, -9.81, 0.0),
        flip_alpha=0.05,
    )
    dem = DEMSystem(
        positions=dem_positions,
        radii=[0.035] * len(dem_positions),
        masses=[1.0] * len(dem_positions),
        damping=0.02,
    )

    coupling = MPDEMCoupling(mpm=mpm, dem=dem, grid_resolution=64)

    def _step():
        coupling.step(
            dt,
            search_radius=0.08,
            k_n=1.5e4,
            c_n=10.0,
            gravity=(0.0, -9.81, 0.0),
            flip_alpha=0.05,
        )

    step_time = timed_step(_step, steps)
    metrics = record_metrics(
        "mpdem_box_sinking", mpm.x, mpm.v, mpm.m, (0.0, -9.81, 0.0), step_time
    )
    print("MPDEM box sinking:", format_metrics(metrics))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", help="Warp execution architecture (cpu/cuda)")
    parser.add_argument("--precision", default="float32", help="Default precision")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=("dem", "mpm", "mpdem"),
        help="Subset of scenarios to run",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    backend_init(args.arch, args.precision)

    selected = set(args.scenarios)
    if "dem" in selected:
        run_dem()
    if "mpm" in selected:
        run_mpm()
    if "mpdem" in selected:
        run_mpdem()


if __name__ == "__main__":
    main()
