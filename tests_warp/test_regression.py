"""Regression tests for GeoWarp solvers against recorded baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

try:  # pragma: no cover - runtime dependency guard
    import warp  # type: ignore
except Exception as exc:  # pragma: no cover - skip when Warp unavailable
    pytest.skip(f"Warp runtime unavailable: {exc}", allow_module_level=True)

_SNAPSHOT_PATH = Path(__file__).with_name("regression_snapshots.json")


def _load_snapshot(name: str) -> dict:
    data = json.loads(_SNAPSHOT_PATH.read_text())
    if name not in data:
        raise KeyError(f"Unknown regression snapshot '{name}'")
    return data[name]


def _vector(values):
    return [float(v) for v in values]


def _norm(vec):
    return math.sqrt(sum(component * component for component in vec))


def _relative_error(reference: float, value: float) -> float:
    scale = max(abs(reference), 1.0e-8)
    return abs(value - reference) / scale


def _vector_error(reference, value) -> float:
    ref_vec = _vector(reference)
    val_vec = _vector(value)
    delta = [a - b for a, b in zip(val_vec, ref_vec)]
    return _norm(delta) / max(_norm(ref_vec), 1.0e-8)


@pytest.mark.parametrize(
    "scenario",
    sorted(json.loads(_SNAPSHOT_PATH.read_text()).keys()),
)
def test_regression_snapshots(scenario: str) -> None:
    snapshot = _load_snapshot(scenario)
    baseline = snapshot["baseline"]
    sample = snapshot.get("current", baseline)
    thresholds = snapshot.get("thresholds", {})

    centroid_error = _vector_error(baseline["centroid"], sample["centroid"])
    assert (
        centroid_error <= thresholds.get("centroid", 0.01)
    ), f"{scenario}: centroid drift {centroid_error:.3%} exceeds tolerance"

    volume_error = _relative_error(baseline["volume"], sample["volume"])
    assert (
        volume_error <= thresholds.get("volume", 0.02)
    ), f"{scenario}: volume error {volume_error:.3%} exceeds tolerance"

    baseline_energy = baseline.get("energy", {})
    sample_energy = sample.get("energy", {})
    max_energy_error = 0.0
    total_reference = 0.0
    total_error = 0.0
    for term, ref_value in baseline_energy.items():
        value = sample_energy.get(term, 0.0)
        total_reference += abs(ref_value)
        total_error += abs(value - ref_value)
        max_energy_error = max(max_energy_error, _relative_error(ref_value, value))
    if total_reference > 0.0:
        aggregate_error = total_error / total_reference
    else:
        aggregate_error = max_energy_error
    assert (
        aggregate_error <= thresholds.get("energy", 0.02)
    ), f"{scenario}: energy error {aggregate_error:.3%} exceeds tolerance"

    baseline_step = baseline.get("step_time", 0.0)
    sample_step = sample.get("step_time", 0.0)
    if baseline_step > 0.0 and sample_step > 0.0:
        slowdown = sample_step / baseline_step - 1.0
        assert (
            slowdown <= thresholds.get("step_time", 0.25)
        ), f"{scenario}: step time regression {slowdown:.3%} exceeds tolerance"
