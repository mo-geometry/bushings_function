import numpy as np
from matplotlib.tri import Triangulation
import scipy.special as sc
from scipy.special import gamma, gammainc, gammaincc, factorial



# OBJECTS ##############################################################################################################

class Surface:
  def __init__(self, config, vertices=44, cigar=False, pancake=False):
    # meshgrids θ + φ
    self.θ, self.φ = np.meshgrid(np.linspace(0, np.pi, vertices).astype('float64'), np.linspace(0, 2 * np.pi, vertices).astype('float64'))
    # generate the path
    # surface meshgrids
    self.r, self.drdθ, self.d2rdθ2 = anisotropicHO(self.θ, η=config["η"], λ=config["λ"], nF=config["nF"],
                                                   cigar=cigar, pancake=pancake, derivatives=None)
    self.x, self.y, self.z, self.tri = delunaySurface(self.r, self.θ, self.φ)
    self.θ0 = np.linspace(0, 2 * np.pi, 20 * vertices)
    # path vectors
    self.r0, self.dr0dθ, self.d2r0dθ2 = anisotropicHO(self.θ0,
                                                      η=config["η"], λ=config["λ"], nF=config["nF"],
                                                      cigar=cigar, pancake=pancake, derivatives=True)
    self.ηλnF = [config["η"], config["λ"], config["nF"]]  # record for display


def anisotropicHO(θ, cigar=True, pancake=False, η=5, λ=11, nF=23, derivatives=None):
    if pancake is True:
        α = (η * np.cos(θ)) ** 2
        β = (η * np.sin(θ)) ** 2
        dαdθ = - 2 * η ** 2 * np.cos(θ) * np.sin(θ)
        dβdθ = 2 * η ** 2 * np.cos(θ) * np.sin(θ)
        d2αdθ2 = - 2 * η ** 2 * np.cos(2 * θ)
        d2βdθ2 = 2 * η ** 2 * np.cos(2 * θ)
    elif cigar is True:
        α = (η * np.sin(θ)) ** 2
        β = (η * np.cos(θ)) ** 2
        dαdθ = 2 * η ** 2 * np.cos(θ) * np.sin(θ)
        dβdθ = - 2 * η ** 2 * np.cos(θ) * np.sin(θ)
        d2αdθ2 = 2 * η ** 2 * np.cos(2 * θ)
        d2βdθ2 = - 2 * η ** 2 * np.cos(2 * θ)
        # polar radius
    # polar radius
    G, dG, d2G = bushinG(α, β, dαdθ, dβdθ, d2αdθ2, d2βdθ2, λ, nF)
    r = gammainc(nF+1, β) + np.exp(-β) * G
    if derivatives is not None:
        nFac = factorial(nF)
        drdθ = np.exp(-β) * (dG - dβdθ * G + (β ** nF * dβdθ) / nFac)
        d2rdθ2 = (np.exp(-β) * (d2G - d2βdθ2 * G - dβdθ * dG
                                + β ** (nF-1) * (nF * dβdθ ** 2 + β * d2βdθ2) / nFac)
                  - dβdθ * drdθ)
    else:
        drdθ = np.zeros(θ.shape)
        d2rdθ2 = np.zeros(θ.shape)
    return r, drdθ, d2rdθ2


def bushinG(α, β, dα, dβ, d2α, d2β, λ, nF, G=0.0, dG=0.0, d2G=0.0):
    np.seterr(divide='ignore', invalid='ignore')  # suppress divide by 0 warning
    for k in range(nF + 1):
        idk = np.floor((nF - k) / λ)
        fack = factorial(k)
        fackidk = (fack * factorial(idk))
        #shorthand
        t1 = β ** k * gammainc(idk + 1, α / λ) / fack
        t2 = β ** k * (α / λ) ** idk * (dα / λ) * np.exp(-α / λ) / fackidk
        t3 = k * (dβ / β) * t1
        #derivatives
        dt1 = t2 + t3
        dt2 = (k * (dβ / β) + idk * (dα / α) - (dα / λ) + (d2α / dα)) * t2
        dt3 = k * (d2β / β) * t1 - k * (dβ / β) ** 2 * t1 + k * (dβ / β) * dt1
        G = G + t1
        dG = dG + dt1
        d2G = d2G + dt2 + dt3
    return G, dG, d2G


# delunay triangulation
def delunaySurface(R, θ, φ):
    # surface
    x = np.ravel(R * np.sin(θ) * np.cos(φ))
    y = np.ravel(R * np.sin(θ) * np.sin(φ))
    z = np.ravel(R * np.cos(θ))
    # delunay triangulation
    return x, y, z,  Triangulation(np.ravel(θ), np.ravel(φ))
