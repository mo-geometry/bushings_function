"""Mesh export helpers (PLY, OBJ).

Provides simple ASCII exporters for polygon meshes with positions, normals,
vertex colours and triangle indices. These are intentionally minimal but
Blender-friendly: PLY preserves per-vertex RGB and normals, OBJ exports
geometry+normals (colours are not standard in OBJ).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_as_ply(
    path: str | Path,
    positions: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Save an indexed triangle mesh as an ASCII PLY file.

    Arguments:
        path: Destination file path.
        positions: (N,3) float32 array
        normals: (N,3) float32 array
        colors: (N,3) float32 array with values in [0,1]
        indices: (M,) uint32 array of triangle indices (0-based, groups of 3)
    """
    p = Path(path)
    verts = int(positions.shape[0])
    faces = int(indices.size // 3)

    with p.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # vertex list
        # convert colours from [0,1] to 0..255 ints
        cols_255 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        for i in range(verts):
            x, y, z = positions[i]
            nx, ny, nz = normals[i]
            r, g, b = cols_255[i]
            f.write(f"{x} {y} {z} {nx} {ny} {nz} {r} {g} {b}\n")

        # faces
        idx = indices.reshape(-1, 3)
        for tri in idx:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def save_as_obj(
    path: str | Path,
    positions: np.ndarray,
    normals: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Save an indexed triangle mesh as a simple OBJ file.

    Notes:
        - OBJ uses 1-based indices.
        - This exporter writes positions and normals; per-vertex colours are
          not supported in standard OBJ and are therefore omitted.
    """
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        f.write("# Exported by bushings_function\n")

        # vertices
        for v in positions:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # normals
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        # faces (use format v//vn)
        idx = indices.reshape(-1, 3)
        for tri in idx:
            # OBJ is 1-based
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
