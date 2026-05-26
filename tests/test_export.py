"""Tests for mesh export (PLY, OBJ) and export UI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from bushings_function.__main__ import _sync_export_extension
from bushings_function.io.export_mesh import save_as_obj, save_as_ply


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def simple_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimal valid mesh: two triangles forming a quad."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    normals = np.tile([0.0, 0.0, 1.0], (4, 1)).astype(np.float32)
    colors = np.tile([0.5, 0.5, 0.5], (4, 1)).astype(np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return positions, normals, colors, indices


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------
class TestSaveAsPly:
    def test_creates_file(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        out = tmp_path / "mesh.ply"
        save_as_ply(out, *simple_mesh)
        assert out.exists()

    def test_ply_header(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        out = tmp_path / "mesh.ply"
        save_as_ply(out, *simple_mesh)
        text = out.read_text(encoding="utf-8")
        assert text.startswith("ply\n")
        assert "format ascii 1.0" in text
        assert "element vertex 4" in text
        assert "element face 2" in text
        assert "end_header" in text

    def test_vertex_count_in_header(self, tmp_path: Path) -> None:
        n = 10
        pos = np.random.rand(n, 3).astype(np.float32)
        nrm = np.zeros((n, 3), dtype=np.float32)
        col = np.ones((n, 3), dtype=np.float32)
        idx = np.zeros(3, dtype=np.uint32)
        out = tmp_path / "mesh.ply"
        save_as_ply(out, pos, nrm, col, idx)
        header = out.read_text(encoding="utf-8")
        assert f"element vertex {n}" in header

    def test_color_clamped_to_255(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        positions, normals, _, indices = simple_mesh
        # Colors slightly outside [0, 1] should be clamped, not wrap-around
        colors = np.array([[1.2, -0.1, 0.5]] * 4, dtype=np.float32)
        out = tmp_path / "mesh.ply"
        save_as_ply(out, positions, normals, colors, indices)
        lines = out.read_text(encoding="utf-8").splitlines()
        header_end = next(i for i, ln in enumerate(lines) if ln == "end_header")
        first_vertex = lines[header_end + 1].split()
        r, g, b = int(first_vertex[6]), int(first_vertex[7]), int(first_vertex[8])
        assert r == 255
        assert g == 0
        assert b == 127

    def test_face_indices_1based_false(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        """PLY uses 0-based indices."""
        out = tmp_path / "mesh.ply"
        save_as_ply(out, *simple_mesh)
        lines = out.read_text(encoding="utf-8").splitlines()
        header_end = next(i for i, ln in enumerate(lines) if ln == "end_header")
        # First face line: "3 0 1 2"
        face_line = lines[header_end + 5]  # 4 vertices, then first face
        parts = face_line.split()
        assert parts[0] == "3"
        assert int(parts[1]) == 0


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------
class TestSaveAsObj:
    def test_creates_file(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        positions, normals, _, indices = simple_mesh
        out = tmp_path / "mesh.obj"
        save_as_obj(out, positions, normals, indices)
        assert out.exists()

    def test_vertex_count(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        positions, normals, _, indices = simple_mesh
        out = tmp_path / "mesh.obj"
        save_as_obj(out, positions, normals, indices)
        lines = out.read_text(encoding="utf-8").splitlines()
        v_lines = [ln for ln in lines if ln.startswith("v ")]
        assert len(v_lines) == 4

    def test_normal_count(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        positions, normals, _, indices = simple_mesh
        out = tmp_path / "mesh.obj"
        save_as_obj(out, positions, normals, indices)
        lines = out.read_text(encoding="utf-8").splitlines()
        vn_lines = [ln for ln in lines if ln.startswith("vn ")]
        assert len(vn_lines) == 4

    def test_face_count(self, tmp_path: Path, simple_mesh: tuple) -> None:  # type: ignore[type-arg]
        positions, normals, _, indices = simple_mesh
        out = tmp_path / "mesh.obj"
        save_as_obj(out, positions, normals, indices)
        lines = out.read_text(encoding="utf-8").splitlines()
        f_lines = [ln for ln in lines if ln.startswith("f ")]
        assert len(f_lines) == 2

    def test_obj_one_based_indices(self, tmp_path: Path, simple_mesh: tuple) -> None:
        """OBJ uses 1-based indices — vertex 0 should appear as '1//1'."""
        positions, normals, _, indices = simple_mesh
        out = tmp_path / "mesh.obj"
        save_as_obj(out, positions, normals, indices)
        lines = out.read_text(encoding="utf-8").splitlines()
        f_lines = [ln for ln in lines if ln.startswith("f ")]
        # First face: indices 0,1,2 → "f 1//1 2//2 3//3"
        assert f_lines[0] == "f 1//1 2//2 3//3"

    def test_accepts_path_string(self, tmp_path: Path, simple_mesh: tuple) -> None:
        """save_as_obj should accept a plain string path."""
        positions, normals, _, indices = simple_mesh
        out = str(tmp_path / "mesh.obj")
        save_as_obj(out, positions, normals, indices)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# Export extension sync (UI logic — no GUI required)
# ---------------------------------------------------------------------------
class TestSyncExportExtension:
    """Tests for the export filename ↔ format synchronisation helper.

    This pattern separates UI state logic from the ImGui widget calls so
    that subtle extension-sync bugs can be caught without a live GL context.
    """

    FORMATS: ClassVar[list[str]] = ["PLY", "OBJ"]

    def test_ply_to_obj(self) -> None:
        assert _sync_export_extension("mesh.ply", 1, self.FORMATS) == "mesh.obj"

    def test_obj_to_ply(self) -> None:
        assert _sync_export_extension("mesh.obj", 0, self.FORMATS) == "mesh.ply"

    def test_stem_preserved(self) -> None:
        assert _sync_export_extension("my_model.ply", 1, self.FORMATS) == "my_model.obj"

    def test_stem_with_dots_preserved(self) -> None:
        """Only the final suffix should change; internal dots are kept."""
        assert _sync_export_extension("v1.2.ply", 1, self.FORMATS) == "v1.2.obj"

    def test_no_extension_input(self) -> None:
        """Filename without extension: stem is the full name."""
        assert _sync_export_extension("mesh", 0, self.FORMATS) == "mesh.ply"

    def test_idempotent(self) -> None:
        """Switching to the already-active format is a no-op."""
        assert _sync_export_extension("mesh.ply", 0, self.FORMATS) == "mesh.ply"
