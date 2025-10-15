"""Shared helpers for GeoWarp regression benchmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import warp as wp

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "tests_warp" / "regression_snapshots.json"


def array_to_numpy(arr: wp.array) -> np.ndarray:
    """Convert a Warp array to a NumPy array on the host."""

    return wp.to_numpy(arr)


def centroid(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return pts.mean(axis=0)


def bounding_box_volume(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1.0e-9)
    return float(np.prod(span))


def kinetic_energy(masses: np.ndarray, velocities: np.ndarray) -> float:
    return float(0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1)))


def potential_energy(
    masses: np.ndarray,
    positions: np.ndarray,
    gravity: Sequence[float] = (0.0, -9.81, 0.0),
    reference_height: float = 0.0,
) -> float:
    gravity_vec = np.asarray(gravity, dtype=np.float64)
    rel = positions - np.array([0.0, reference_height, 0.0], dtype=np.float64)
    return float(-np.sum(masses * rel.dot(gravity_vec)))


def update_snapshot(name: str, metrics: Mapping[str, object]) -> None:
    """Update the recorded metrics for ``name`` with the provided ``metrics``."""

    data = json.loads(SNAPSHOT_PATH.read_text())
    scenario = data.setdefault(name, {})
    scenario["current"] = dict(metrics)
    scenario.setdefault("baseline", dict(metrics))
    scenario.setdefault(
        "thresholds",
        {"centroid": 0.01, "volume": 0.02, "energy": 0.02, "step_time": 0.25},
    )
    SNAPSHOT_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def timed_step(step_fn, count: int) -> float:
    """Execute ``step_fn`` ``count`` times and return the average seconds per step."""

    start = time.perf_counter()
    for _ in range(count):
        step_fn()
        wp.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / max(count, 1)


def metrics_from_particles(
    positions: wp.array,
    velocities: wp.array,
    masses: wp.array,
    gravity: Sequence[float],
) -> Dict[str, object]:
    pos_np = array_to_numpy(positions)
    vel_np = array_to_numpy(velocities)
    mass_np = array_to_numpy(masses)

    return {
        "centroid": centroid(pos_np).tolist(),
        "volume": bounding_box_volume(pos_np),
        "energy": {
            "kinetic": kinetic_energy(mass_np, vel_np),
            "potential": potential_energy(mass_np, pos_np, gravity),
        },
    }


def record_metrics(
    name: str,
    positions: wp.array,
    velocities: wp.array,
    masses: wp.array,
    gravity: Sequence[float],
    step_time: float,
) -> Dict[str, object]:
    metrics = metrics_from_particles(positions, velocities, masses, gravity)
    metrics["step_time"] = step_time
    update_snapshot(name, metrics)
    return metrics


def format_metrics(metrics: Mapping[str, object]) -> str:
    centroid_vec = ", ".join(f"{v:.4f}" for v in metrics["centroid"])
    energy = metrics["energy"]
    energy_terms = ", ".join(f"{k}={v:.4f}" for k, v in energy.items())
    return (
        f"centroid=({centroid_vec}), volume={metrics['volume']:.4f}, "
        f"step_time={metrics['step_time']:.6f}, energy[{energy_terms}]"
    )

__all__ = [
    "array_to_numpy",
    "metrics_from_particles",
    "record_metrics",
    "timed_step",
    "format_metrics",
]
