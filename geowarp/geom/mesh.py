"""Triangle mesh utilities built on top of Warp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import warp as wp


@dataclass
class TriangleMesh:
    """Lightweight wrapper around :class:`warp.Mesh` with BVH support."""

    vertices: Sequence[Sequence[float]]
    indices: Sequence[Sequence[int]]
    device: Optional[str] = None
    build_bvh: bool = True

    def __post_init__(self) -> None:
        dev = self.device or wp.get_device()
        self.device = dev
        self.vertex_array = wp.array(self.vertices, dtype=wp.vec3f, device=dev)
        self.index_array = wp.array(self.indices, dtype=wp.vec3i, device=dev)
        try:
            self.mesh = wp.Mesh(points=self.vertex_array, indices=self.index_array)
        except TypeError:
            self.mesh = wp.Mesh(vertices=self.vertex_array, indices=self.index_array)
        self.bvh = wp.BVH(self.mesh) if self.build_bvh else None

    @classmethod
    def from_obj(
        cls,
        path: str | Path,
        device: Optional[str] = None,
        build_bvh: bool = True,
    ) -> "TriangleMesh":
        """Load a triangle mesh from an OBJ file using Warp's importer."""

        mesh = wp.Mesh.load(str(path))
        vertices = getattr(mesh, "points", getattr(mesh, "vertices")).numpy()
        triangles = getattr(mesh, "indices", getattr(mesh, "triangles")).numpy()
        return cls(vertices=vertices, indices=triangles, device=device, build_bvh=build_bvh)

    @property
    def mesh_id(self) -> wp.uint64:
        """Return the Warp mesh handle for kernel launches."""

        return self.mesh.id

    @property
    def bvh_id(self) -> Optional[wp.uint64]:
        """Return the BVH handle when available."""

        return self.bvh.id if self.bvh is not None else None

    def update_vertices(self, vertices: Iterable[Sequence[float]]) -> None:
        """Update vertex positions in-place and refit the BVH."""

        wp.copy(self.vertex_array, wp.array(vertices, dtype=wp.vec3f, device=self.device))
        if self.bvh is not None:
            self.bvh.refit(self.mesh)

    def ensure_bvh(self) -> wp.BVH:
        """Ensure a BVH is built for fast intersection queries."""

        if self.bvh is None:
            self.bvh = wp.BVH(self.mesh)
        return self.bvh

