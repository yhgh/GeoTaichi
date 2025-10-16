"""Generate representative GeoWarp visualizations for demo purposes.

The script normally executes the lightweight DEM, MPM, and MPDEM regression
scenarios and stores their particle/grid states as VTK files inside the chosen
output directory. When the Warp runtime is unavailable (for instance on
documentation builds or CPU-only CI agents) a deterministic *stub* dataset is
generated instead so downstream consumers always find a ready-to-visualise set
of files under ``demo/outputs``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - import guard for environments without Warp
    from geowarp.backend import init as backend_init

    from bench_warp.run_regression import run_dem, run_mpdem, run_mpm
except Exception as exc:  # pragma: no cover - exercised in CPU-only CI
    backend_init = None
    run_dem = run_mpdem = run_mpm = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when Warp is available
    _IMPORT_ERROR = None

import math
import random


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
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Force stub VTK generation even when Warp is available. Useful for "
            "documentation or quick smoke tests."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(args.scenarios)

    if args.stub or backend_init is None:
        _generate_stub_outputs(output_root, requested)
        return

    backend_init(args.arch, args.precision)

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


def _generate_stub_outputs(output_root: Path, scenarios: set[str]) -> None:
    """Emit deterministic VTK placeholders when Warp is unavailable."""

    if _IMPORT_ERROR is not None:
        message = (
            "Warp runtime unavailable ({}); generating stub demo outputs instead."
        ).format(_IMPORT_ERROR)
    else:
        message = "Generating stub demo outputs as requested."
    print(message)

    if "dem" in scenarios:
        _write_stub_dem(output_root / "dem")

    if "mpm" in scenarios:
        _write_stub_mpm(output_root / "mpm")

    if "mpdem" in scenarios:
        _write_stub_mpdem(output_root / "mpdem")


def _write_stub_dem(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    n_layers = 4
    particles_per_layer = 24
    radii = [0.1 + (0.3 - 0.1) * i / (particles_per_layer - 1) for i in range(particles_per_layer)]
    theta = [2.0 * math.pi * i / particles_per_layer for i in range(particles_per_layer)]
    layers = [0.6 * lid / max(1, n_layers - 1) for lid in range(n_layers)]

    points = []
    velocities = []
    ids = []
    for lid, z in enumerate(layers):
        for pid, (r, t) in enumerate(zip(radii, theta)):
            x = r * math.cos(t)
            y = r * math.sin(t)
            points.append((x, y, z))
            velocities.append((-
                y * 0.5,
                x * 0.5,
                0.1 * lid,
            ))
            ids.append(lid * particles_per_layer + pid)

    _write_ascii_vtu(
        folder / "frame_0000.vtu",
        points,
        {
            "id": ids,
            "velocity": velocities,
        },
    )

    speeds = [_vector_magnitude(v) for v in velocities]
    if speeds:
        s_min = min(speeds)
        s_max = max(speeds)
        denom = s_max - s_min if s_max > s_min else 1.0
        speeds = [(s - s_min) / denom for s in speeds]
    _write_svg(
        folder / "frame_0000.svg",
        *_build_particle_svg(
            points,
            speeds,
            value_to_color=_speed_to_color,
        ),
    )


def _write_stub_mpm(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    dims = (8, 6, 1)
    origin = (0.0, 0.0, 0.0)
    spacing = (0.1, 0.1, 0.5)

    nx, ny, nz = dims
    density = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                density.append(1.0 + (i + j + k) / max(1, nx + ny + nz - 3))

    velocity = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                velocity.append((0.0, 0.0, (i + j + k) / max(1, nx + ny + nz)))

    _write_ascii_vts(
        folder / "frame_0000.vts",
        dims,
        origin,
        spacing,
        cell_fields={"density": density},
        point_fields={"velocity": velocity},
    )

    _write_svg(
        folder / "frame_0000.svg",
        *_build_grid_svg(density, dims),
    )


def _write_stub_mpdem(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    n_particles = 32
    rng = random.Random(2024)
    positions = [
        (
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.1, 0.1),
        )
        for _ in range(n_particles)
    ]
    velocities = [
        (
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.1, 0.1),
        )
        for _ in range(n_particles)
    ]
    phases = [rng.randint(0, 1) for _ in range(n_particles)]

    _write_ascii_vtu(
        folder / "frame_0000.vtu",
        positions,
        {
            "velocity": velocities,
            "phase": phases,
        },
    )

    _write_svg(
        folder / "frame_0000.svg",
        *_build_particle_svg(
            positions,
            phases,
            value_to_color=_phase_to_color,
        ),
    )


def _write_ascii_vtu(path: Path, points, point_fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    npoints = len(points)
    point_strings = "\n          ".join(
        f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points
    )
    point_data_blocks = []
    for name, values in point_fields.items():
        if not values:
            continue
        first = values[0]
        if isinstance(first, (tuple, list)):
            comps = len(first)
            dtype = "Float32"
            body = "\n          ".join(
                " ".join(f"{float(c):.6f}" for c in entry) for entry in values
            )
        else:
            comps = 1
            if isinstance(first, float):
                dtype = "Float32"
                body = "\n          ".join(f"{float(v):.6f}" for v in values)
            else:
                dtype = "Int32"
                body = "\n          ".join(str(int(v)) for v in values)
        point_data_blocks.append(
            (
                f'<DataArray type="{dtype}" Name="{name}" '
                f'NumberOfComponents="{comps}" format="ascii">\n'
                f"          {body}\n        </DataArray>"
            )
        )

    connectivity = " ".join(str(i) for i in range(npoints))
    offsets = " ".join(str(i + 1) for i in range(npoints))
    cell_types = " ".join("1" for _ in range(npoints))  # VTK_VERTEX

    point_data_section = "\n        ".join(point_data_blocks)

    xml = f"""<?xml version=\"1.0\"?>
<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">
  <UnstructuredGrid>
    <Piece NumberOfPoints=\"{npoints}\" NumberOfCells=\"{npoints}\">
      <PointData>
        {point_data_section}
      </PointData>
      <CellData/>
      <Points>
        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">
          {point_strings}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">
          {connectivity}
        </DataArray>
        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">
          {offsets}
        </DataArray>
        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">
          {cell_types}
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    path.write_text(xml)


def _write_ascii_vts(
    path: Path,
    dims,
    origin,
    spacing,
    *,
    cell_fields,
    point_fields,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = dims
    ox, oy, oz = origin
    sx, sy, sz = spacing

    points = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                points.append(
                    (
                        ox + i * sx,
                        oy + j * sy,
                        oz + k * sz,
                    )
                )

    point_strings = "\n          ".join(
        f"{x:.6f} {y:.6f} {z:.6f}" for (x, y, z) in points
    )

    def _format_field_blocks(data_map):
        blocks = []
        for name, values in data_map.items():
            if not values:
                continue
            first = values[0]
            if isinstance(first, (tuple, list)):
                comps = len(first)
                dtype = "Float32"
                body = "\n          ".join(
                    " ".join(f"{float(c):.6f}" for c in entry) for entry in values
                )
            else:
                comps = 1
                if isinstance(first, float):
                    dtype = "Float32"
                    body = "\n          ".join(f"{float(v):.6f}" for v in values)
                else:
                    dtype = "Int32"
                    body = "\n          ".join(str(int(v)) for v in values)
            blocks.append(
                (
                    f'<DataArray type="{dtype}" Name="{name}" '
                    f'NumberOfComponents="{comps}" format="ascii">\n'
                    f"          {body}\n        </DataArray>"
                )
            )
        return blocks

    point_data_blocks = _format_field_blocks(point_fields)
    cell_data_blocks = _format_field_blocks(cell_fields)
    point_data_section = "\n        ".join(point_data_blocks)
    cell_data_section = "\n        ".join(cell_data_blocks)

    xml = f"""<?xml version=\"1.0\"?>
<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">
  <StructuredGrid WholeExtent=\"0 {nx} 0 {ny} 0 {nz}\">
    <Piece Extent=\"0 {nx} 0 {ny} 0 {nz}\">
      <PointData>
        {point_data_section}
      </PointData>
      <CellData>
        {cell_data_section}
      </CellData>
      <Points>
        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">
          {point_strings}
        </DataArray>
      </Points>
    </Piece>
  </StructuredGrid>
</VTKFile>
"""

    path.write_text(xml)


def _build_particle_svg(points, values, *, value_to_color, image_size=512):
    """Return SVG geometry representing particle previews."""

    width = height = image_size
    elements = []

    if not points:
        return width, height, [
            '<rect x="0" y="0" width="{0}" height="{0}" fill="#181a1a" />'.format(
                image_size
            )
        ]

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    extent_x = max(max_x - min_x, 1e-6)
    extent_y = max(max_y - min_y, 1e-6)

    margin = max(12, image_size // 16)
    usable_w = max(image_size - 2 * margin, 1)
    usable_h = max(image_size - 2 * margin, 1)
    radius = max(3.0, image_size / 120.0)

    elements.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#12141a" />'
    )

    for idx, point in enumerate(points):
        x, y, _z = point
        px = margin + (x - min_x) / extent_x * (usable_w - 1)
        py = margin + (y - min_y) / extent_y * (usable_h - 1)
        py = height - py

        color = _rgb_to_hex(value_to_color(values[idx]))
        elements.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{color}" />'
        )

    return width, height, elements


def _build_grid_svg(density_values, dims, *, base_size=512):
    nx, ny, nz = dims

    if nx <= 0 or ny <= 0:
        return base_size, base_size, [
            '<rect x="0" y="0" width="{0}" height="{0}" fill="#181a20" />'.format(
                base_size
            )
        ]

    per_cell = max(6, min(40, base_size // max(nx, ny, 1)))
    margin = per_cell
    width = nx * per_cell + 2 * margin
    height = ny * per_cell + 2 * margin

    values = []
    for j in range(ny):
        row = []
        for i in range(nx):
            accum = 0.0
            for k in range(nz):
                idx = k * ny * nx + j * nx + i
                accum += float(density_values[idx])
            row.append(accum / max(nz, 1))
        values.append(row)

    flat = [v for row in values for v in row]
    v_min = min(flat)
    v_max = max(flat)
    denom = v_max - v_min if v_max > v_min else 1.0

    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#101218" />'
    ]

    for j, row in enumerate(values):
        base_y = margin + j * per_cell
        for i, value in enumerate(row):
            weight = (value - v_min) / denom
            color = _rgb_to_hex(_speed_to_color(weight))
            base_x = margin + i * per_cell
            elements.append(
                f'<rect x="{base_x}" y="{height - base_y - per_cell}" width="{per_cell}" '
                f'height="{per_cell}" fill="{color}" />'
            )

    return width, height, elements


def _vector_magnitude(vec) -> float:
    if isinstance(vec, (list, tuple)) and len(vec) >= 3:
        x, y, z = vec[0], vec[1], vec[2]
    else:
        x = y = z = float(vec)
    return math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))


def _speed_to_color(value: float):
    t = max(0.0, min(1.0, float(value)))
    if t < 0.5:
        return _lerp_color((58, 110, 165), (142, 202, 230), t * 2.0)
    return _lerp_color((142, 202, 230), (244, 138, 54), (t - 0.5) * 2.0)


def _phase_to_color(phase):
    return (244, 138, 54) if int(round(float(phase))) else (76, 201, 240)


def _lerp_color(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(
        int(round(a_c + (b_c - a_c) * t)) for a_c, b_c in zip(a, b)
    )


def _rgb_to_hex(color) -> str:
    r, g, b = (int(color[0]) & 0xFF, int(color[1]) & 0xFF, int(color[2]) & 0xFF)
    return f"#{r:02x}{g:02x}{b:02x}"


def _write_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
    ]
    content.extend(elements)
    content.append("</svg>")
    path.write_text("\n".join(content))


if __name__ == "__main__":
    main()
