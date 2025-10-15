"""Generate representative GeoWarp visualizations for demo purposes.

This script runs the lightweight DEM, MPM, and MPDEM scenarios used in the
regression suite and saves their particle states as VTK files under the chosen
output directory. The produced files can be opened directly in ParaView to
verify visual parity with the original GeoTaichi examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from geowarp.backend import init as backend_init

from bench_warp.run_regression import run_dem, run_mpdem, run_mpm


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        default="cpu",
        help="Warp execution architecture (cpu or cuda)",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        help="Default floating point precision (float32 or float64)",
    )
    parser.add_argument(
        "--output",
        default="demo/outputs",
        help="Destination directory for generated VTK files",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Number of steps between saved frames",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=("dem", "mpm", "mpdem"),
        help="Subset of scenarios to visualise",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    backend_init(args.arch, args.precision)

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(args.scenarios)

    if "dem" in requested:
        run_dem(
            frame_dir=output_root / "dem",
            frame_interval=args.frame_interval,
        )

    if "mpm" in requested:
        run_mpm(
            frame_dir=output_root / "mpm",
            frame_interval=args.frame_interval,
        )

    if "mpdem" in requested:
        run_mpdem(
            frame_dir=output_root / "mpdem",
            frame_interval=max(1, args.frame_interval),
        )


if __name__ == "__main__":
    main()
