"""Execute lightweight GeoWarp regression scenarios and update baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

from geowarp.backend import init as backend_init
from geowarp.dem.core import DEMSystem
from geowarp.mpm.explicit import ExplicitMPMSolver
from geowarp.mpdem.coupling import MPDEMCoupling

from geowarp.io_vtk import write_points_vtu

from .common import array_to_numpy, format_metrics, record_metrics, timed_step


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


def run_dem(
    dt: float = 2.5e-4,
    steps: int = 20,
    *,
    frame_dir: Optional[Path] = None,
    frame_interval: int = 5,
) -> None:
    positions = _grid_positions((6, 4, 1), (0.05, 0.05, 0.05), (0.0, 0.05, 0.0))
    radii = [0.0225] * len(positions)
    masses = [0.8] * len(positions)

    dem = DEMSystem(positions=positions, radii=radii, masses=masses, damping=0.05)

    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    def _step():
        dem.step(dt, search_radius=0.06, k_n=2.5e4, c_n=15.0)

    def _write_frame(step_index: int) -> None:
        if frame_dir is None:
            return
        if step_index % frame_interval != 0 and step_index != steps - 1:
            return
        write_points_vtu(
            frame_dir / f"dem_column_collapse_{step_index:04d}",
            array_to_numpy(dem.x),
            point_data={
                "velocity": array_to_numpy(dem.v),
                "radius": array_to_numpy(dem.r),
            },
        )

    step_time = timed_step(_step, steps, frame_callback=_write_frame)
    metrics = record_metrics(
        "dem_column_collapse", dem.x, dem.v, dem.m, (0.0, -9.81, 0.0), step_time
    )
    print("DEM column collapse:", format_metrics(metrics))


def run_mpm(
    dt: float = 1.0e-3,
    steps: int = 15,
    *,
    frame_dir: Optional[Path] = None,
    frame_interval: int = 5,
) -> None:
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

    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    def _step():
        solver.step(dt)

    def _write_frame(step_index: int) -> None:
        if frame_dir is None:
            return
        if step_index % frame_interval != 0 and step_index != steps - 1:
            return
        write_points_vtu(
            frame_dir / f"mpm_column_collapse_{step_index:04d}",
            array_to_numpy(solver.x),
            point_data={
                "velocity": array_to_numpy(solver.v),
                "mass": array_to_numpy(solver.m),
            },
        )

    step_time = timed_step(_step, steps, frame_callback=_write_frame)
    metrics = record_metrics(
        "mpm_column_collapse", solver.x, solver.v, solver.m, (0.0, -9.81, 0.0), step_time
    )
    print("MPM column collapse:", format_metrics(metrics))


def run_mpdem(
    dt: float = 7.5e-4,
    steps: int = 12,
    *,
    frame_dir: Optional[Path] = None,
    frame_interval: int = 4,
) -> None:
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

    if frame_dir is not None:
        (frame_dir / "mpm").mkdir(parents=True, exist_ok=True)
        (frame_dir / "dem").mkdir(parents=True, exist_ok=True)

    def _step():
        coupling.step(
            dt,
            search_radius=0.08,
            k_n=1.5e4,
            c_n=10.0,
            gravity=(0.0, -9.81, 0.0),
            flip_alpha=0.05,
        )

    def _write_frame(step_index: int) -> None:
        if frame_dir is None:
            return
        if step_index % frame_interval != 0 and step_index != steps - 1:
            return
        write_points_vtu(
            frame_dir / "mpm" / f"mpdem_mpm_{step_index:04d}",
            array_to_numpy(mpm.x),
            point_data={
                "velocity": array_to_numpy(mpm.v),
                "mass": array_to_numpy(mpm.m),
            },
        )
        write_points_vtu(
            frame_dir / "dem" / f"mpdem_dem_{step_index:04d}",
            array_to_numpy(dem.x),
            point_data={
                "velocity": array_to_numpy(dem.v),
                "radius": array_to_numpy(dem.r),
            },
        )

    step_time = timed_step(_step, steps, frame_callback=_write_frame)
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
    parser.add_argument(
        "--frame-dir",
        type=str,
        default=None,
        help="Optional directory for saving VTK frames during execution",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Number of steps between saved frames",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    backend_init(args.arch, args.precision)

    selected = set(args.scenarios)
    frame_root = Path(args.frame_dir).resolve() if args.frame_dir else None
    if "dem" in selected:
        run_dem(
            frame_dir=frame_root / "dem" if frame_root else None,
            frame_interval=args.frame_interval,
        )
    if "mpm" in selected:
        run_mpm(
            frame_dir=frame_root / "mpm" if frame_root else None,
            frame_interval=args.frame_interval,
        )
    if "mpdem" in selected:
        run_mpdem(
            frame_dir=frame_root / "mpdem" if frame_root else None,
            frame_interval=max(1, args.frame_interval),
        )


if __name__ == "__main__":
    main()


__all__ = ["run_dem", "run_mpm", "run_mpdem", "main"]
