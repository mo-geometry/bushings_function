"""Surface of revolution — Delaunay triangulation and coordinate transforms.

Generates 3D mesh data from a Bushing's function radial profile by revolving
r(θ) around the polar axis and triangulating with matplotlib's Delaunay
triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.tri import Triangulation  # type: ignore[import-untyped]

from bushings_function.core.bushings import BushingsProfile, SurfaceType, evaluate_bushings

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
FloatArray = npt.NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Dataclass for surface mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SurfaceOfRevolution:
    """Triangulated surface of revolution in Cartesian coordinates.

    Attributes:
        x: X coordinates of vertices, shape (M,).
        y: Y coordinates of vertices, shape (M,).
        z: Z coordinates of vertices, shape (M,).
        triangulation: Matplotlib Triangulation object for rendering.
        profile: The BushingsProfile used to generate this surface.
    """

    x: FloatArray
    y: FloatArray
    z: FloatArray
    triangulation: Triangulation  # type: ignore[type-arg]
    profile: BushingsProfile


# ---------------------------------------------------------------------------
# Surface generation
# ---------------------------------------------------------------------------
def surface_to_cartesian(
    r: FloatArray,
    theta: FloatArray,
    phi: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert polar surface-of-revolution coordinates to Cartesian.

    Args:
        r: Radial values, shape matching theta meshgrid.
        theta: Polar angle meshgrid.
        phi: Azimuthal angle meshgrid.

    Returns:
        Tuple (x, y, z) as ravelled 1D arrays.
    """
    x = np.ravel(r * np.sin(theta) * np.cos(phi))
    y = np.ravel(r * np.sin(theta) * np.sin(phi))
    z = np.ravel(r * np.cos(theta))
    return x, y, z


def create_surface_mesh(
    eta: float = 5.0,
    lam: int = 11,
    n_f: int = 23,
    surface_type: SurfaceType = SurfaceType.PANCAKE,
    vertices: int = 44,
    d_theta: float = 1e-16,
) -> SurfaceOfRevolution:
    """Generate a triangulated surface of revolution from Bushing's function.

    Creates a meshgrid in (θ, φ), evaluates r(θ) via Bushing's function,
    converts to Cartesian coordinates, and applies Delaunay triangulation.

    Args:
        eta: Gaussian spread parameter η.
        lam: Anisotropy parameter λ.
        n_f: Occupancy parameter nF.
        surface_type: Cigar or pancake oscillator orientation.
        vertices: Number of grid points along each angular axis.
        d_theta: Small offset to avoid division by zero at poles.

    Returns:
        SurfaceOfRevolution with mesh data and triangulation.
    """
    # Create meshgrid
    theta_1d = np.linspace(d_theta, np.pi - d_theta, vertices)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, vertices)
    theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d)

    # Evaluate radial profile on flattened grid
    theta_flat = theta_grid.flatten()
    profile = evaluate_bushings(
        theta_flat, eta=eta, lam=lam, n_f=n_f, surface_type=surface_type,
    )

    # Reshape r back to grid for Cartesian conversion
    r_grid = profile.r.reshape(theta_grid.shape)
    x, y, z = surface_to_cartesian(r_grid, theta_grid, phi_grid)

    # Delaunay triangulation on the angular coordinates
    tri = Triangulation(np.ravel(theta_grid), np.ravel(phi_grid))

    return SurfaceOfRevolution(
        x=x,
        y=y,
        z=z,
        triangulation=tri,
        profile=profile,
    )
