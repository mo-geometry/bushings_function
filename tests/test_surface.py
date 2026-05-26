"""Tests for the surface geometry module."""

from __future__ import annotations

import numpy as np

from bushings_function.core.bushings import SurfaceType
from bushings_function.geometry.surface import (
    SurfaceOfRevolution,
    create_surface_mesh,
    surface_to_cartesian,
)


class TestSurfaceToCartesian:
    """Coordinate transform tests."""

    def test_unit_sphere(self) -> None:
        """r=1 should produce points on the unit sphere."""
        theta = np.linspace(0.1, np.pi - 0.1, 20)
        phi = np.linspace(0.0, 2 * np.pi, 20)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        r = np.ones_like(theta_grid)
        x, y, z = surface_to_cartesian(r, theta_grid, phi_grid)
        radii = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-10)

    def test_output_shape(self) -> None:
        """Output should be ravelled 1D arrays."""
        n = 10
        theta = np.linspace(0.1, 3.0, n)
        phi = np.linspace(0.0, 6.0, n)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        r = np.ones_like(theta_grid) * 2.0
        x, _y, _z = surface_to_cartesian(r, theta_grid, phi_grid)
        assert x.ndim == 1
        assert x.shape[0] == n * n

    def test_equator_circle(self) -> None:
        """At θ=π/2 with r=1, x² + y² should equal 1."""
        phi = np.linspace(0.0, 2 * np.pi, 100)
        theta = np.full_like(phi, np.pi / 2)
        r = np.ones_like(phi)
        x, y, z = surface_to_cartesian(r, theta, phi)
        np.testing.assert_allclose(x**2 + y**2, 1.0, atol=1e-10)
        np.testing.assert_allclose(z, 0.0, atol=1e-10)


class TestCreateSurfaceMesh:
    """Surface mesh generation tests."""

    def test_returns_surface_of_revolution(self) -> None:
        mesh = create_surface_mesh(vertices=10)
        assert isinstance(mesh, SurfaceOfRevolution)

    def test_mesh_has_triangulation(self) -> None:
        mesh = create_surface_mesh(vertices=10)
        assert mesh.triangulation is not None
        assert len(mesh.triangulation.triangles) > 0

    def test_vertex_count(self) -> None:
        """Number of vertices should be vertices²."""
        n = 12
        mesh = create_surface_mesh(vertices=n)
        assert mesh.x.shape[0] == n * n

    def test_both_surface_types(self) -> None:
        """Both cigar and pancake should produce valid meshes."""
        for st in SurfaceType:
            mesh = create_surface_mesh(vertices=10, surface_type=st)
            assert not np.any(np.isnan(mesh.x))
            assert not np.any(np.isnan(mesh.y))
            assert not np.any(np.isnan(mesh.z))

    def test_profile_attached(self) -> None:
        """The mesh should carry the profile used to generate it."""
        mesh = create_surface_mesh(eta=3.0, lam=5, n_f=10, vertices=10)
        assert mesh.profile.eta == 3.0
        assert mesh.profile.lam == 5
        assert mesh.profile.n_f == 10

    def test_surface_bounded(self) -> None:
        """Surface vertices should be bounded by the max radial value."""
        mesh = create_surface_mesh(vertices=20)
        r_max = float(np.max(mesh.profile.r)) + 1e-6
        radii = np.sqrt(mesh.x**2 + mesh.y**2 + mesh.z**2)
        assert np.all(radii <= r_max + 1e-6)
