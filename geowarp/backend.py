"""Warp backend compatibility layer for GeoWarp."""
from __future__ import annotations

import warp as wp

_DEFAULT_DTYPE = wp.float32


def init(arch: str = "cuda", precision: str = "float32") -> dict[str, object]:
    """Initialize the Warp backend for GeoWarp usage.

    Args:
        arch: Target execution architecture, accepting "cuda", "gpu", or "cpu".
        precision: Floating point precision selector, supporting "float32"/"float64"
            and aliases like "double".

    Returns:
        A dictionary capturing the selected Warp device and default dtype.
    """

    dev = "cuda" if arch in ("cuda", "gpu") else "cpu"
    wp.set_device(dev)

    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = wp.float64 if precision in ("float64", "double") else wp.float32
    return {"device": wp.get_device(), "dtype": _DEFAULT_DTYPE}


__all__ = ["init"]
