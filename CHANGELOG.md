# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-26

### Added

**Project scaffolding**
- `pyproject.toml` with hatchling build backend, src layout, optional `[gpu]` extras
- GitHub Actions CI/CD: ruff lint, mypy typecheck, pytest matrix (Python 3.11/3.12/3.13)
- Pre-commit hooks: ruff (lint + format), mypy strict mode
- Makefile with `test`, `lint`, `format`, `typecheck` targets

**Core math** (`core/bushings.py`)
- `evaluate_bushings()` — surface-of-revolution radial profile r(θ) and optional
  first/second derivatives dr/dθ, d²r/dθ²
- `bushings_incomplete_gamma()` — incomplete gamma series Σ β^k/k! · γ*(⌊(nF−k)/λ⌋+1, α/λ)
- `anisotropic_ho_params()` — cigar/pancake α(θ), β(θ) mappings and their derivatives
- `BushingsProfile` frozen dataclass; `SurfaceType` enum (CIGAR, PANCAKE, SPHERE)

**Geometry** (`geometry/surface.py`)
- `SurfaceOfRevolution` dataclass for Delaunay-triangulated mesh data
- `create_surface_mesh()` — meshgrid evaluation, Delaunay triangulation via matplotlib
- `surface_to_cartesian()` — polar SOR → Cartesian coordinates

**Geometric transforms** (`core/transforms.py`)
- `euler_to_rotation()` — ZYX Euler angles → 3×3 rotation matrix
- `axis_angle_to_rotation()` — Rodrigues' formula
- `look_at()` — 4×4 view matrix

**Orbit camera** (`core/orbit_camera.py`)
- `OrbitCamera` — arcball-style mouse orbit with zoom; GLFW callback interface

**GPU-ready mesh generation** (`assets/sor_mesh.py`)
- `create_sor_mesh()` — UV-parameterised mesh with analytic normals (∂P/∂θ × ∂P/∂φ)
- Four colour schemes: uniform, theta gradient, radius gradient, height gradient

**Ground-plane grid** (`assets/grid.py`)
- `create_grid()` — XZ-plane line grid with highlighted X/Z axes

**OpenGL rendering** (`rendering/`)
- `gl_utils.py` — shader compilation, program linking, VAO/VBO/EBO creation,
  uniform upload helpers (`set_uniform_mat4`, `set_uniform_vec3`, `set_uniform_float`,
  `set_uniform_int`)
- `shaders.py` — GLSL 3.30 Blinn-Phong vertex/fragment shaders (4-light array,
  per-vertex colour, alpha) and line shaders for grid/wireframe

**Interactive renderer** (`__main__.py`)
- GLFW 3.3 core-profile window (1280×720, MSAA ×4, vsync)
- Dear ImGui overlay panels: surface type, Bushing's parameters (η, λ, nF),
  colour scheme, alpha, mesh density (slices/stacks), lighting (4 lights,
  enable/disable, position, colour)
- Middle-mouse to toggle ImGui; left-drag to orbit; scroll to zoom
- JSON config persistence (`~/.bushings_function_config.json`) — changes survive
  across sessions

**Mesh export** (`io/export_mesh.py`)
- `save_as_ply()` — ASCII PLY with positions, normals, and 8-bit RGB vertex colours
- `save_as_obj()` — OBJ with geometry and normals (v//vn format, 1-based indices)
- Export panel in ImGui: format selector (PLY/OBJ), filename input (extension
  auto-syncs on format change), export button with status feedback

**Test suite** (`tests/`)
- `test_bushings.py` — parameter validation, α/β symmetry identities, r(θ) bounds,
  analytical derivatives vs. finite-difference verification
- `test_surface.py` — unit-sphere Cartesian transform, vertex count, NaN checks,
  profile attachment
- `test_export.py` — PLY/OBJ file creation, header correctness, index conventions,
  colour clamping; `_sync_export_extension` unit tests (no GL context required)

[Unreleased]: https://github.com/mo-geometry/bushings_function/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mo-geometry/bushings_function/releases/tag/v0.1.0
