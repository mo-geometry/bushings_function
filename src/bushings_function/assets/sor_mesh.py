"""Surface-of-revolution mesh generation for OpenGL rendering.

Generates GPU-ready vertex data (positions, normals, colours, triangle indices)
from a Bushing's function radial profile. The mesh is built using the same UV
parameterisation as the sphere template but with radius r(θ) from
evaluate_bushings().
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bushings_function.core.bushings import SurfaceType, evaluate_bushings


def _generate_colors(
    positions: npt.NDArray[np.float32],
    scheme: str,
    stacks: int,
    slices: int,
    base_color: tuple[float, float, float] = (0.85, 0.55, 0.25),
) -> npt.NDArray[np.float32]:
    """Generate per-vertex colors based on a color scheme.

    Args:
        positions: (N, 3) vertex positions.
        scheme: Color scheme name ("uniform", "theta_gradient", "radius_gradient",
            "height_gradient").
        stacks: Number of latitude divisions.
        slices: Number of longitude divisions.
        base_color: Base RGB color for uniform scheme.

    Returns:
        (N, 3) float32 color array.
    """
    n_verts = positions.shape[0]
    colors = np.zeros((n_verts, 3), dtype=np.float32)

    if scheme == "uniform":
        colors[:] = base_color
    elif scheme == "theta_gradient":
        # Blue (north pole) → red (south pole), one value per stack ring
        t = np.repeat(
            np.linspace(0.0, 1.0, stacks + 1, dtype=np.float32),
            slices + 1,
        )
        colors[:, 0] = t
        colors[:, 2] = 1.0 - t
    elif scheme == "radius_gradient":
        # Dark blue (axis) → bright cyan (equatorial edge)
        radii = np.sqrt(positions[:, 0] ** 2 + positions[:, 2] ** 2)
        r_max = float(radii.max())
        if r_max > 0:
            t = (radii / r_max).astype(np.float32)
        else:
            t = np.zeros(n_verts, dtype=np.float32)
        colors[:, 1] = t
        colors[:, 2] = 1.0
    elif scheme == "height_gradient":
        # Dark purple (bottom) → bright yellow (top)
        heights = positions[:, 1]
        h_min, h_max = float(heights.min()), float(heights.max())
        h_range = h_max - h_min if h_max > h_min else 1.0
        t = ((heights - h_min) / h_range).astype(np.float32)
        colors[:, 0] = t
        colors[:, 1] = 0.5 * t
        colors[:, 2] = 1.0 - t
    else:
        colors[:] = base_color

    return colors


def create_sor_mesh(
    eta: float = 5.0,
    lam: int = 11,
    n_f: int = 23,
    surface_type: SurfaceType = SurfaceType.PANCAKE,
    slices: int = 64,
    stacks: int = 32,
    color: tuple[float, float, float] = (0.85, 0.55, 0.25),
    color_scheme: str = "uniform",
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
]:
    """Generate a surface-of-revolution mesh from Bushing's function.

    Builds a UV mesh where each latitude ring has radius r(θ) from
    evaluate_bushings(). The mesh uses the same layout as create_uv_sphere
    so it plugs directly into the same VAO/shader setup.

    Args:
        eta: Gaussian spread parameter η.
        lam: Anisotropy parameter λ.
        n_f: Occupancy parameter nF.
        surface_type: Cigar or pancake oscillator orientation.
        slices: Number of longitudinal divisions.
        stacks: Number of latitudinal divisions.
        color: RGB colour for the surface (used in "uniform" scheme).
        color_scheme: Color scheme to apply ("uniform", "theta_gradient",
            "radius_gradient", "height_gradient").

    Returns:
        Tuple of (positions, normals, colors, indices) where:
        - positions: (N, 3) float32 vertex positions
        - normals: (N, 3) float32 per-vertex normals
        - colors: (N, 3) float32 per-vertex RGB colours
        - indices: (M,) uint32 triangle indices
    """
    # Evaluate r(θ) along the meridian, with derivatives for normals
    d_theta = 1e-12
    theta_1d = np.linspace(d_theta, np.pi - d_theta, stacks + 1, dtype=np.float64)
    if surface_type == SurfaceType.SPHERE:
        r_vals = np.ones_like(theta_1d, dtype=np.float64)
        dr_vals = np.zeros_like(theta_1d, dtype=np.float64)
    else:
        profile = evaluate_bushings(
            theta_1d, eta=eta, lam=lam, n_f=n_f,
            surface_type=surface_type, derivatives=True,
        )
        r_vals = profile.r
        dr_vals = profile.dr

    positions = []
    normals = []

    for i in range(stacks + 1):
        theta = theta_1d[i]
        r = float(r_vals[i])
        dr = float(dr_vals[i])

        for j in range(slices + 1):
            phi = 2.0 * np.pi * j / slices

            # Surface of revolution: x = r sinθ cosφ, y = r cosθ, z = r sinθ sinφ
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)

            x = r * sin_t * cos_p
            y = r * cos_t
            z = r * sin_t * sin_p
            positions.append([x, y, z])

            # Normal via cross product of tangent vectors:
            # ∂P/∂θ = (dr sinθ + r cosθ) cosφ,  (dr cosθ - r sinθ),  (dr sinθ + r cosθ) sinφ
            # ∂P/∂φ = -r sinθ sinφ,  0,  r sinθ cosφ
            #
            # Outward normal = ∂P/∂θ × ∂P/∂φ (unnormalised)
            dt_rho = dr * sin_t + r * cos_t  # d(r sinθ)/dθ
            dt_z = dr * cos_t - r * sin_t     # d(r cosθ)/dθ
            r_sin_t = r * sin_t

            # Normal = ∂P/∂θ × ∂P/∂φ
            dp_dt = np.array([dt_rho * cos_p, dt_z, dt_rho * sin_p])
            dp_dp = np.array([-r_sin_t * sin_p, 0.0, r_sin_t * cos_p])
            n = np.cross(dp_dt, dp_dp)

            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-12 else np.array([sin_t * cos_p, cos_t, sin_t * sin_p])

            normals.append(n.tolist())

    pos_array = np.array(positions, dtype=np.float32)
    norm_array = np.array(normals, dtype=np.float32)

    # Generate colors based on selected scheme
    color_array = _generate_colors(
        pos_array, color_scheme, stacks, slices, base_color=color
    )

    # Build triangle indices (same topology as UV sphere)
    indices = []
    for i in range(stacks):
        for j in range(slices):
            first = i * (slices + 1) + j
            second = first + slices + 1
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    idx_array = np.array(indices, dtype=np.uint32)

    return pos_array, norm_array, color_array, idx_array
