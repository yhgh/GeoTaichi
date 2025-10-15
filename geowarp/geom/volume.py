"""Level-set volume helpers backed by Warp NanoVDB volumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import warp as wp


@dataclass
class SignedDistanceVolume:
    """Wrapper for Warp volumes storing signed distance fields."""

    volume: wp.Volume
    device: Optional[str] = None

    def __post_init__(self) -> None:
        dev = self.device or wp.get_device()
        self.device = dev
        # Warp volumes live on the GPU already but we keep a handle for kernels.
        self.volume = self.volume

    @classmethod
    def from_nano_vdb(
        cls,
        path: str | Path,
        device: Optional[str] = None,
    ) -> "SignedDistanceVolume":
        """Load a NanoVDB grid into a Warp volume."""

        volume = wp.Volume.load(str(path))
        return cls(volume=volume, device=device)

    @property
    def volume_id(self) -> wp.uint64:
        """Return the Warp volume identifier for kernels."""

        return self.volume.id

    def sample(self, position: Sequence[float]) -> float:
        """Sample the SDF at a world-space position."""

        pos = wp.vec3f(*position)
        return wp.volume_sample_world(self.volume.id, pos)

