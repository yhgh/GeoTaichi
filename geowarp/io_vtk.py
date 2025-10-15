"""GeoWarp-compatible VTK output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from third_party.pyevtk.hl import gridToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkVertex

ScalarLike = Union[np.ndarray, Sequence[float]]
VectorLike = Union[Tuple[ScalarLike, ...], Sequence[ScalarLike], np.ndarray]
FieldMapping = Mapping[str, Union[ScalarLike, VectorLike]]


def write_points_vtu(
    filename: Union[str, Path],
    points: Union[np.ndarray, Sequence[Sequence[float]]],
    point_data: Optional[FieldMapping] = None,
    *,
    cell_data: Optional[FieldMapping] = None,
    field_data: Optional[FieldMapping] = None,
) -> str:
    """Write a point cloud as an unstructured grid (``.vtu`` file).

    ``points`` may contain two or three coordinates per entry; 2-D inputs are
    padded with zeros for the z-component. ``point_data`` and ``cell_data`` may
    contain scalar arrays with shape ``(N,)`` or ``(N, 1)`` or vector-valued
    arrays with shape ``(N, 2)``/``(N, 3)``. Vector arrays are split into the
    component tuples expected by PyEVTK. Tuple/list values are passed through
    after enforcing contiguity.
    """

    base, _ = _normalize_vtk_path(filename, ".vtu")

    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("points must be a (N, 2) or (N, 3) array-like of coordinates")

    npoints = coords.shape[0]
    xyz = _pad_points(coords)

    point_payload = _normalize_unstructured_fields(point_data, npoints)
    cell_payload = _normalize_unstructured_fields(cell_data, npoints)
    field_payload = _normalize_field_data(field_data)

    connectivity = np.arange(npoints, dtype=np.int32)
    offsets = np.arange(1, npoints + 1, dtype=np.int32)
    cell_types = np.full(npoints, VtkVertex.tid, dtype=np.uint8)

    return unstructuredGridToVTK(
        base,
        xyz[0],
        xyz[1],
        xyz[2],
        connectivity,
        offsets,
        cell_types,
        cellData=cell_payload,
        pointData=point_payload,
        fieldData=field_payload,
    )


def write_grid_vts(
    filename: Union[str, Path],
    dimensions: Optional[Sequence[int]] = None,
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    coordinates: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    cell_data: Optional[FieldMapping] = None,
    point_data: Optional[FieldMapping] = None,
    field_data: Optional[FieldMapping] = None,
) -> str:
    """Write a rectilinear/structured grid (``.vts`` file).

    Parameters mirror the legacy GeoTaichi helpers. ``dimensions`` specifies the
    number of cells along each axis. When present, ``origin`` and ``spacing`` are
    used to synthesise the grid nodes. Supplying ``coordinates`` overrides this
    behaviour and allows callers to provide explicit node locations (1-D arrays
    for rectilinear grids or 3-D arrays for logically structured grids).
    """

    base, _ = _normalize_vtk_path(filename, ".vts")

    if coordinates is not None:
        if len(coordinates) != 3:
            raise ValueError("coordinates must be a tuple of three arrays")
        x, y, z = (np.ascontiguousarray(np.asarray(axis)) for axis in coordinates)
    else:
        if dimensions is None:
            raise ValueError("dimensions are required when coordinates are not provided")
        nx, ny, nz = _ensure_triple(dimensions)
        ox, oy, oz = _ensure_triple(origin, pad_value=0.0)
        sx, sy, sz = _ensure_triple(spacing, pad_value=1.0)

        x = ox + sx * np.arange(nx + 1, dtype=np.float64)
        y = oy + sy * np.arange(ny + 1, dtype=np.float64)
        z = oz + sz * np.arange(nz + 1, dtype=np.float64)

    cell_payload = _normalize_grid_fields(cell_data)
    point_payload = _normalize_grid_fields(point_data)
    field_payload = _normalize_field_data(field_data)

    return gridToVTK(
        base,
        x,
        y,
        z,
        cellData=cell_payload,
        pointData=point_payload,
        fieldData=field_payload,
    )


def _normalize_vtk_path(path: Union[str, Path], extension: str) -> Tuple[str, str]:
    p = Path(path)
    if p.suffix:
        if p.suffix != extension:
            p = p.with_suffix("")
        else:
            p = p.with_suffix("")
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return str(p), str(p.with_suffix(extension))


def _ensure_triple(values: Sequence[Union[int, float]], pad_value: float = 0.0) -> Tuple[float, float, float]:
    if len(values) == 3:
        return tuple(float(v) for v in values)
    if len(values) == 2:
        return (float(values[0]), float(values[1]), float(pad_value))
    raise ValueError("Expected a sequence of length 2 or 3")


def _pad_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(points[:, 0])
    y = np.ascontiguousarray(points[:, 1])
    if points.shape[1] == 3:
        z = np.ascontiguousarray(points[:, 2])
    else:
        z = np.zeros(points.shape[0], dtype=points.dtype)
    return x, y, z


def _normalize_unstructured_fields(
    data: Optional[FieldMapping],
    expected_length: int,
) -> Optional[MutableMapping[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]]:
    if not data:
        return None

    normalized: MutableMapping[str, Union[np.ndarray, Tuple[np.ndarray, ...]]] = {}
    for name, value in data.items():
        normalized[name] = _normalize_unstructured_value(name, value, expected_length)
    return normalized


def _normalize_unstructured_value(
    name: str,
    value: Union[ScalarLike, VectorLike],
    expected_length: int,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if isinstance(value, (tuple, list)):
        components = tuple(
            _ensure_1d_component(name, np.asarray(component), expected_length)
            for component in value
        )
        if len(components) == 1:
            return components[0]
        return components

    array = np.asarray(value)
    if array.ndim == 1:
        if array.size != expected_length:
            raise ValueError(
                f"Field '{name}' must have length {expected_length}, got {array.size}"
            )
        return np.ascontiguousarray(array)

    if array.ndim == 2 and array.shape[0] == expected_length:
        if array.shape[1] == 1:
            return np.ascontiguousarray(array[:, 0])
        if array.shape[1] in (2, 3):
            comps = [np.ascontiguousarray(array[:, i]) for i in range(array.shape[1])]
            if array.shape[1] == 2:
                comps.append(np.zeros(expected_length, dtype=array.dtype))
            return tuple(comps)

    raise ValueError(
        f"Field '{name}' must be 1D of length {expected_length} or a (N, 2)/(N, 3) array"
    )


def _ensure_1d_component(name: str, component: np.ndarray, expected_length: int) -> np.ndarray:
    flat = np.ravel(component)
    if flat.size != expected_length:
        raise ValueError(
            f"Component of field '{name}' must have length {expected_length}, got {flat.size}"
        )
    return np.ascontiguousarray(flat)


def _normalize_grid_fields(
    data: Optional[FieldMapping],
) -> Optional[MutableMapping[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]]:
    if not data:
        return None

    normalized: MutableMapping[str, Union[np.ndarray, Tuple[np.ndarray, ...]]] = {}
    for name, value in data.items():
        if isinstance(value, (tuple, list)):
            normalized[name] = tuple(
                np.ascontiguousarray(np.asarray(component)) for component in value
            )
        else:
            normalized[name] = np.ascontiguousarray(np.asarray(value))
    return normalized


def _normalize_field_data(
    data: Optional[FieldMapping],
) -> Optional[MutableMapping[str, np.ndarray]]:
    if not data:
        return None
    normalized: MutableMapping[str, np.ndarray] = {}
    for name, value in data.items():
        normalized[name] = np.ascontiguousarray(np.asarray(value))
    return normalized
