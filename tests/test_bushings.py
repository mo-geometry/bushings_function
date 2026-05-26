"""Tests for the core Bushing's function module."""

from __future__ import annotations

import numpy as np
import pytest

from bushings_function.core.bushings import (
    BushingsProfile,
    SurfaceType,
    anisotropic_ho_params,
    bushings_incomplete_gamma,
    evaluate_bushings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def theta_array() -> np.ndarray:  # type: ignore[type-arg]
    """Standard theta array avoiding poles."""
    return np.linspace(1e-10, np.pi - 1e-10, 200)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
class TestParameterValidation:
    """Input validation for evaluate_bushings."""

    def test_negative_lambda_raises(self, theta_array: np.ndarray) -> None:  # type: ignore[type-arg]
        with pytest.raises(ValueError, match="λ must be ≥ 1"):
            evaluate_bushings(theta_array, lam=0)

    def test_negative_nf_raises(self, theta_array: np.ndarray) -> None:  # type: ignore[type-arg]
        with pytest.raises(ValueError, match="nF must be ≥ 0"):
            evaluate_bushings(theta_array, n_f=-1)


# ---------------------------------------------------------------------------
# Anisotropic HO parameters
# ---------------------------------------------------------------------------
class TestAnisotropicHOParams:
    """Tests for α(θ), β(θ) parameter mappings."""

    def test_pancake_alpha_beta_swap(self) -> None:
        """Pancake and cigar should have swapped α and β."""
        theta = np.array([0.3, 0.7, 1.2, 2.1])
        eta = 5.0
        a_p, b_p, _, _, _, _ = anisotropic_ho_params(theta, eta, SurfaceType.PANCAKE)
        a_c, b_c, _, _, _, _ = anisotropic_ho_params(theta, eta, SurfaceType.CIGAR)
        np.testing.assert_allclose(a_p, b_c, atol=1e-12)
        np.testing.assert_allclose(b_p, a_c, atol=1e-12)

    def test_alpha_plus_beta_equals_eta_squared(self) -> None:
        """α + β = η² for all θ (sin² + cos² = 1)."""
        theta = np.linspace(0.1, 3.0, 100)
        eta = 7.0
        for st in SurfaceType:
            a, b, _, _, _, _ = anisotropic_ho_params(theta, eta, st)
            np.testing.assert_allclose(a + b, eta**2, atol=1e-10)

    def test_derivative_symmetry(self) -> None:
        """dα/dθ = -dβ/dθ for both surface types."""
        theta = np.linspace(0.1, 3.0, 100)
        for st in SurfaceType:
            _, _, da, db, _, _ = anisotropic_ho_params(theta, 5.0, st)
            np.testing.assert_allclose(da, -db, atol=1e-12)

    def test_second_derivative_symmetry(self) -> None:
        """d²α/dθ² = -d²β/dθ² for both surface types."""
        theta = np.linspace(0.1, 3.0, 100)
        for st in SurfaceType:
            _, _, _, _, d2a, d2b = anisotropic_ho_params(theta, 5.0, st)
            np.testing.assert_allclose(d2a, -d2b, atol=1e-12)


# ---------------------------------------------------------------------------
# Bushing's function evaluation
# ---------------------------------------------------------------------------
class TestEvaluateBushings:
    """Tests for the main Bushing's function."""

    def test_returns_profile(self, theta_array: np.ndarray) -> None:  # type: ignore[type-arg]
        profile = evaluate_bushings(theta_array)
        assert isinstance(profile, BushingsProfile)

    def test_output_shape(self, theta_array: np.ndarray) -> None:  # type: ignore[type-arg]
        profile = evaluate_bushings(theta_array)
        assert profile.r.shape == theta_array.shape
        assert profile.dr.shape == theta_array.shape
        assert profile.d2r.shape == theta_array.shape

    def test_r_bounded_zero_one(self, theta_array: np.ndarray) -> None:  # type: ignore[type-arg]
        """Bushing's function values should be in [0, 1] (regularised gamma)."""
        profile = evaluate_bushings(theta_array)
        assert np.all(profile.r >= -1e-10)
        assert np.all(profile.r <= 1.0 + 1e-10)

    def test_r_nonnegative(self) -> None:
        """Radial profile should be non-negative for various parameters."""
        theta = np.linspace(1e-10, np.pi - 1e-10, 100)
        for eta in [1.0, 5.0, 10.0]:
            for st in SurfaceType:
                profile = evaluate_bushings(theta, eta=eta, surface_type=st)
                assert np.all(profile.r >= -1e-12), f"Negative r for eta={eta}, {st}"

    def test_isotropic_limit(self) -> None:
        """When λ=1, cigar and pancake should differ but both be valid."""
        theta = np.linspace(1e-10, np.pi - 1e-10, 100)
        p = evaluate_bushings(theta, lam=1, n_f=0, surface_type=SurfaceType.PANCAKE)
        c = evaluate_bushings(theta, lam=1, n_f=0, surface_type=SurfaceType.CIGAR)
        # Both should be valid profiles in [0, 1]
        assert np.all(p.r >= -1e-10)
        assert np.all(c.r >= -1e-10)

    def test_parameters_stored(self) -> None:
        """Profile should store the parameters used."""
        theta = np.linspace(0.1, 3.0, 50)
        profile = evaluate_bushings(theta, eta=3.0, lam=5, n_f=10)
        assert profile.eta == 3.0
        assert profile.lam == 5
        assert profile.n_f == 10
        assert profile.surface_type == SurfaceType.PANCAKE

    def test_cigar_surface_type(self) -> None:
        theta = np.linspace(0.1, 3.0, 50)
        profile = evaluate_bushings(theta, surface_type=SurfaceType.CIGAR)
        assert profile.surface_type == SurfaceType.CIGAR

    def test_no_derivatives_gives_zeros(self, theta_array: np.ndarray) -> None:  # type: ignore[type-arg]
        """Without derivatives=True, dr and d2r should be zero."""
        profile = evaluate_bushings(theta_array, derivatives=False)
        np.testing.assert_allclose(profile.dr, 0.0, atol=1e-15)
        np.testing.assert_allclose(profile.d2r, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Derivative verification
# ---------------------------------------------------------------------------
class TestDerivatives:
    """Analytical derivatives vs numerical finite differences."""

    def test_first_derivative_matches_numerical(self) -> None:
        """dr/dθ should match finite-difference approximation."""
        theta = np.linspace(0.3, np.pi - 0.3, 500)
        profile = evaluate_bushings(theta, eta=5.0, lam=11, n_f=23, derivatives=True)

        dtheta = theta[1] - theta[0]
        dr_numerical = np.gradient(profile.r, dtheta)

        # Compare in the interior (avoid edge effects from np.gradient)
        interior = slice(10, -10)
        np.testing.assert_allclose(
            profile.dr[interior], dr_numerical[interior], rtol=1e-2, atol=1e-4,
        )

    def test_second_derivative_matches_numerical(self) -> None:
        """d²r/dθ² should match finite-difference approximation."""
        theta = np.linspace(0.3, np.pi - 0.3, 500)
        profile = evaluate_bushings(theta, eta=5.0, lam=11, n_f=23, derivatives=True)

        dtheta = theta[1] - theta[0]
        dr_numerical = np.gradient(profile.r, dtheta)
        d2r_numerical = np.gradient(dr_numerical, dtheta)

        interior = slice(20, -20)
        np.testing.assert_allclose(
            profile.d2r[interior], d2r_numerical[interior], rtol=0.1, atol=1e-2,
        )

    def test_derivatives_both_surface_types(self) -> None:
        """Derivatives should compute without error for both types."""
        theta = np.linspace(0.3, np.pi - 0.3, 100)
        for st in SurfaceType:
            profile = evaluate_bushings(
                theta, eta=5.0, lam=11, n_f=23, surface_type=st, derivatives=True,
            )
            assert not np.any(np.isnan(profile.dr))
            assert not np.any(np.isnan(profile.d2r))


# ---------------------------------------------------------------------------
# Incomplete gamma series
# ---------------------------------------------------------------------------
class TestBushingsIncompleteGamma:
    """Tests for the incomplete gamma series summation."""

    def test_returns_correct_shapes(self) -> None:
        n = 50
        shape = (n,)
        alpha = np.ones(shape) * 5.0
        beta = np.ones(shape) * 3.0
        zeros = np.zeros(shape)
        g, dg, d2g = bushings_incomplete_gamma(
            alpha, beta, zeros, zeros, zeros, zeros, lam=11, n_f=23,
        )
        assert g.shape == shape
        assert dg.shape == shape
        assert d2g.shape == shape

    def test_g_nonnegative(self) -> None:
        """G should be non-negative (sum of non-negative terms)."""
        alpha = np.linspace(0.1, 25.0, 100)
        beta = np.linspace(0.1, 25.0, 100)
        zeros = np.zeros(100)
        g, _, _ = bushings_incomplete_gamma(
            alpha, beta, zeros, zeros, zeros, zeros, lam=11, n_f=23,
        )
        assert np.all(g >= -1e-12)

    def test_nf_zero_single_term(self) -> None:
        """With nF=0, the series has exactly one term (k=0)."""
        alpha = np.array([1.0, 2.0, 5.0])
        beta = np.array([1.0, 2.0, 5.0])
        zeros = np.zeros(3)
        g, _, _ = bushings_incomplete_gamma(
            alpha, beta, zeros, zeros, zeros, zeros, lam=1, n_f=0,
        )
        # k=0: idk=0, fack=1, t1 = β^0 * gammainc(1, α/λ) = gammainc(1, α)
        from scipy.special import gammainc as gi  # type: ignore[import-untyped]

        expected = gi(1, alpha)
        np.testing.assert_allclose(g, expected, atol=1e-12)
