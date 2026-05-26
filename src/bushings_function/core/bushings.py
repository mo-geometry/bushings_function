"""Bushing's function — pure NumPy/SciPy, no UI dependencies.

Evaluates the radial profile of surfaces of revolution derived from the
coherent states of the anisotropic harmonic oscillator. The function is
parameterised by (η, λ, nF) — Gaussian spread, anisotropy, and Fermi
occupancy respectively.

The core evaluation uses the regularised incomplete gamma function from
SciPy and a summation over the upper incomplete gamma series.

References:
    Busch et al., Europhys. Lett. 44, 1 (1998).
    O'Sullivan & Busch, Phys. Rev. A 79, 033602 (2009).
    O'Sullivan, PhD thesis (2012), §2.3, Eq. (2.112).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import factorial, gammainc  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
FloatArray = npt.NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Surface type enumeration
# ---------------------------------------------------------------------------
class SurfaceType(Enum):
    """Orientation of the anisotropic harmonic oscillator."""

    CIGAR = "cigar"
    PANCAKE = "pancake"
    SPHERE = "sphere"


# ---------------------------------------------------------------------------
# Dataclass for evaluated profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BushingsProfile:
    """Radial profile of a Bushing's surface of revolution.

    Attributes:
        theta: Polar angle array, shape (N,).
        r: Radial profile r(θ), shape (N,).
        dr: First derivative dr/dθ, shape (N,). Zero if derivatives=False.
        d2r: Second derivative d²r/dθ², shape (N,). Zero if derivatives=False.
        surface_type: Which oscillator orientation produced this profile.
        eta: Gaussian spread parameter used.
        lam: Anisotropy parameter used.
        n_f: Occupancy parameter used.
    """

    theta: FloatArray
    r: FloatArray
    dr: FloatArray
    d2r: FloatArray
    surface_type: SurfaceType
    eta: float
    lam: int
    n_f: int


# ---------------------------------------------------------------------------
# Anisotropic HO parameter mappings
# ---------------------------------------------------------------------------
def anisotropic_ho_params(
    theta: FloatArray,
    eta: float,
    surface_type: SurfaceType,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute α(θ), β(θ) and their first/second derivatives.

    For the cigar oscillator:
        α(θ) = η²sin²(θ),  β(θ) = η²cos²(θ)
    For the pancake oscillator:
        α(θ) = η²cos²(θ),  β(θ) = η²sin²(θ)

    Args:
        theta: Polar angle array, shape (N,).
        eta: Gaussian spread parameter.
        surface_type: Cigar or pancake orientation.

    Returns:
        Tuple (α, β, dα/dθ, dβ/dθ, d²α/dθ², d²β/dθ²).
    """
    eta2 = eta**2
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_2t = np.cos(2.0 * theta)
    sin_cos = cos_t * sin_t

    if surface_type == SurfaceType.PANCAKE:
        alpha = eta2 * cos_t**2
        beta = eta2 * sin_t**2
        d_alpha = -2.0 * eta2 * sin_cos
        d_beta = 2.0 * eta2 * sin_cos
        d2_alpha = -2.0 * eta2 * cos_2t
        d2_beta = 2.0 * eta2 * cos_2t
    else:  # CIGAR
        alpha = eta2 * sin_t**2
        beta = eta2 * cos_t**2
        d_alpha = 2.0 * eta2 * sin_cos
        d_beta = -2.0 * eta2 * sin_cos
        d2_alpha = 2.0 * eta2 * cos_2t
        d2_beta = -2.0 * eta2 * cos_2t

    return alpha, beta, d_alpha, d_beta, d2_alpha, d2_beta


# ---------------------------------------------------------------------------
# Incomplete gamma series
# ---------------------------------------------------------------------------
def bushings_incomplete_gamma(
    alpha: FloatArray,
    beta: FloatArray,
    d_alpha: FloatArray,
    d_beta: FloatArray,
    d2_alpha: FloatArray,
    d2_beta: FloatArray,
    lam: int,
    n_f: int,
    *,
    derivatives: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate the incomplete gamma series in Bushing's function.

    This computes the summation:
        G = Σ_{k=0}^{nF} β^k / k! · γ*(⌊(nF-k)/λ⌋ + 1, α/λ)

    and optionally its first and second derivatives with respect to θ.

    Args:
        alpha: α(θ) array, shape (N,).
        beta: β(θ) array, shape (N,).
        d_alpha: dα/dθ array, shape (N,).
        d_beta: dβ/dθ array, shape (N,).
        d2_alpha: d²α/dθ² array, shape (N,).
        d2_beta: d²β/dθ² array, shape (N,).
        lam: Anisotropy parameter λ.
        n_f: Occupancy parameter nF.
        derivatives: Whether to compute first and second derivatives.

    Returns:
        Tuple (G, dG/dθ, d²G/dθ²). Derivatives are zero arrays if
        derivatives=False.
    """
    shape = alpha.shape
    g_sum = np.zeros(shape)
    dg_sum = np.zeros(shape)
    d2g_sum = np.zeros(shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(n_f + 1):
            idk = int(np.floor((n_f - k) / lam))
            fack = float(factorial(k, exact=True))

            # Core term: β^k / k! · γ*(idk+1, α/λ)
            t1 = beta**k * gammainc(idk + 1, alpha / lam) / fack
            g_sum = g_sum + t1

            if derivatives:
                fack_idk = fack * float(factorial(idk, exact=True))
                # Term from differentiating the incomplete gamma
                t2 = (
                    beta**k
                    * (alpha / lam) ** idk
                    * (d_alpha / lam)
                    * np.exp(-alpha / lam)
                    / fack_idk
                )
                # Term from differentiating β^k
                t3 = k * (d_beta / beta) * t1

                dt1 = t2 + t3
                dt2 = (
                    k * (d_beta / beta)
                    + idk * (d_alpha / alpha)
                    - (d_alpha / lam)
                    + (d2_alpha / d_alpha)
                ) * t2
                dt3 = (
                    k * (d2_beta / beta) * t1
                    - k * (d_beta / beta) ** 2 * t1
                    + k * (d_beta / beta) * dt1
                )
                dg_sum = dg_sum + dt1
                d2g_sum = d2g_sum + dt2 + dt3

    if not derivatives:
        dg_sum = np.zeros(shape)
        d2g_sum = np.zeros(shape)

    return g_sum, dg_sum, d2g_sum


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_bushings(
    theta: FloatArray,
    eta: float = 5.0,
    lam: int = 11,
    n_f: int = 23,
    surface_type: SurfaceType = SurfaceType.PANCAKE,
    *,
    derivatives: bool = False,
) -> BushingsProfile:
    """Evaluate the Bushing's function radial profile.

    Computes r(θ) = γ*(nF+1, β) + exp(-β) · G(α, β, λ, nF) where G is the
    incomplete gamma series, and optionally the first and second derivatives.

    Args:
        theta: Polar angle array, shape (N,). Should avoid exact 0 and π
            to prevent division-by-zero in derivatives.
        eta: Gaussian spread parameter η.
        lam: Anisotropy parameter λ (positive integer).
        n_f: Occupancy parameter nF (non-negative integer).
        surface_type: Cigar or pancake oscillator orientation.
        derivatives: Whether to compute dr/dθ and d²r/dθ².

    Returns:
        BushingsProfile with the evaluated radial profile and derivatives.

    Raises:
        ValueError: If lam < 1 or n_f < 0.
    """
    if lam < 1:
        msg = f"Anisotropy parameter λ must be ≥ 1, got {lam}"
        raise ValueError(msg)
    if n_f < 0:
        msg = f"Occupancy parameter nF must be ≥ 0, got {n_f}"
        raise ValueError(msg)

    alpha, beta, d_alpha, d_beta, d2_alpha, d2_beta = anisotropic_ho_params(
        theta, eta, surface_type
    )

    g, dg, d2g = bushings_incomplete_gamma(
        alpha, beta, d_alpha, d_beta, d2_alpha, d2_beta,
        lam, n_f, derivatives=derivatives,
    )

    # r(θ) = γ*(nF+1, β) + exp(-β) · G
    r = gammainc(n_f + 1, beta) + np.exp(-beta) * g

    if derivatives:
        n_fac = float(factorial(n_f, exact=True))
        dr = np.exp(-beta) * (dg - d_beta * g + (beta**n_f * d_beta) / n_fac)
        d2r = (
            np.exp(-beta) * (
                d2g - d2_beta * g - d_beta * dg
                + beta ** (n_f - 1)
                * (n_f * d_beta**2 + beta * d2_beta)
                / n_fac
            )
            - d_beta * dr
        )
    else:
        dr = np.zeros_like(theta)
        d2r = np.zeros_like(theta)

    return BushingsProfile(
        theta=theta,
        r=r,
        dr=dr,
        d2r=d2r,
        surface_type=surface_type,
        eta=eta,
        lam=lam,
        n_f=n_f,
    )
