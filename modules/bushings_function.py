import numpy as np
from matplotlib.tri import Triangulation
import scipy.special as sc
from scipy.special import gamma, gammainc, gammaincc, factorial



# OBJECTS ##############################################################################################################

class BUSHINGS:
  def __init__(self, η=5.0, λ=11, nF=23, vertices=44, sphere=False, derivatives=False, dθ=1e-16):
    self.settings = {"η": η, "λ": λ, "nF": nF}
    self.type = ["pancake", "cigar"]
    # meshgrids θ + φ [shift by 1e-16 to avoid division by 0 in Bushings derivatives]
    θ = np.linspace(dθ, np.pi - dθ, vertices).astype('float64')
    φ = np.linspace(0, 2 * np.pi, vertices).astype('float64')
    θ, φ = np.meshgrid(θ, φ)
    self.surface = self.anisotropicHO(θ, sphere=sphere, derivatives=derivatives)
    # 2D profile
    self.plane = self.anisotropicHO(np.linspace(dθ, 2 * np.pi - dθ, 20 * vertices), sphere=sphere, derivatives=True)
    # surface meshgrids
    self.delunaySurface(θ, φ, sphere=sphere)


  # BUSHINGS FUNCTION: SURFACES OF REVOLUTION
  def anisotropicHO(self, θ, derivatives=None, sphere=False):
      if sphere:
          self.settings = {"η": 10, "λ": 1, "nF": 0}
          self.type = ["sphere"]
          return {"sphere": {"θ": θ, "r": np.ones(θ.shape), "dr": np.zeros(θ.shape), "d2r": np.zeros(θ.shape)}}
      if len(θ.shape)==2:
          h_w, θ = θ.shape, θ.flatten()
      else:
          h_w = len(θ)
      η, λ, nF = self.settings["η"], self.settings["λ"], self.settings["nF"]
      surface = {"pancake": {}, "cigar": {}}
      for surf in list(surface.keys()):
          if surf=="pancake":
              α = (η * np.cos(θ)) ** 2
              β = (η * np.sin(θ)) ** 2
              dαdθ = - 2 * η ** 2 * np.cos(θ) * np.sin(θ)
              dβdθ = 2 * η ** 2 * np.cos(θ) * np.sin(θ)
              d2αdθ2 = - 2 * η ** 2 * np.cos(2 * θ)
              d2βdθ2 = 2 * η ** 2 * np.cos(2 * θ)
          elif surf=="cigar":
              α = (η * np.sin(θ)) ** 2
              β = (η * np.cos(θ)) ** 2
              dαdθ = 2 * η ** 2 * np.cos(θ) * np.sin(θ)
              dβdθ = - 2 * η ** 2 * np.cos(θ) * np.sin(θ)
              d2αdθ2 = 2 * η ** 2 * np.cos(2 * θ)
              d2βdθ2 = - 2 * η ** 2 * np.cos(2 * θ)
          variables = {"α": α, "β": β, "dα": dαdθ, "dβ": dβdθ, "d2α": d2αdθ2, "d2β": d2βdθ2}
          # Evaluate the upper incomplete gamma function portion of Bushings function
          G, dG, d2G = self.upper_incomplete_bushings(variables, λ, nF, derivatives=derivatives)
          r = gammainc(nF + 1, β) + np.exp(-β) * G
          if derivatives:
              nFac = factorial(nF)
              drdθ = np.exp(-β) * (dG - dβdθ * G + (β ** nF * dβdθ) / nFac)
              d2rdθ2 = (np.exp(-β) * (d2G - d2βdθ2 * G - dβdθ * dG
                                      + β ** (nF-1) * (nF * dβdθ ** 2 + β * d2βdθ2) / nFac)
                        - dβdθ * drdθ)
          else:
              drdθ = np.zeros(θ.shape)
              d2rdθ2 = np.zeros(θ.shape)
          # add to dictionary
          surface[surf] = {"θ": θ.reshape(h_w),
                           "r": r.reshape(h_w), "dr": drdθ.reshape(h_w), "d2r": d2rdθ2.reshape(h_w)}
      return surface


  @staticmethod
  def upper_incomplete_bushings(variables, λ, nF, G=0.0, dG=0.0, d2G=0.0, derivatives=False):
      α, β = variables["α"], variables["β"]
      dα, dβ = variables["dα"], variables["dβ"]
      d2α, d2β = variables["d2α"], variables["d2β"]
      np.seterr(divide='ignore', invalid='ignore')  # suppress divide by 0 warning
      for k in range(nF + 1):
          idk = np.floor((nF - k) / λ)
          fack = factorial(k)
          # shorthand
          t1 = β ** k * gammainc(idk + 1, α / λ) / fack
          # fill
          G = G + t1
          if derivatives:
              fackidk = (fack * factorial(idk))
              t2 = β ** k * (α / λ) ** idk * (dα / λ) * np.exp(-α / λ) / fackidk
              t3 = k * (dβ / β) * t1
              # derivatives
              dt1 = t2 + t3
              dt2 = (k * (dβ / β) + idk * (dα / α) - (dα / λ) + (d2α / dα)) * t2
              dt3 = k * (d2β / β) * t1 - k * (dβ / β) ** 2 * t1 + k * (dβ / β) * dt1
              dG = dG + dt1
              d2G = d2G + dt2 + dt3
      if derivatives is not True:
          dG, d2G = np.zeros(G.shape), np.zeros(G.shape)
      return G, dG, d2G


  # delunay triangulation
  def delunaySurface(self, θ, φ, sphere=False):
      if sphere:
          # surface
          self.surface["sphere"]["x"] = np.ravel(np.sin(θ) * np.cos(φ))
          self.surface["sphere"]["y"] = np.ravel(np.sin(θ) * np.sin(φ))
          self.surface["sphere"]["z"] = np.ravel(np.cos(θ))
          # delunay triangulation
          # add triangulation to dictionary
          self.surface["sphere"]["tri"] = Triangulation(np.ravel(θ), np.ravel(φ))
      elif sphere is None: # if not plotting - save computation time by omitting Delunay Triangulation
          # surface
          self.surface["sphere"]["x"] = -1
          self.surface["sphere"]["y"] = -1
          self.surface["sphere"]["z"] = -1
          # delunay triangulation
          # add triangulation to dictionary
          self.surface["sphere"]["tri"] = -1
      else:
          # else Bushings surfaces of revolution
          for surf in list(self.surface.keys()):
              # surface
              self.surface[surf]["x"] = np.ravel(self.surface[surf]["r"] * np.sin(θ) * np.cos(φ))
              self.surface[surf]["y"] = np.ravel(self.surface[surf]["r"] * np.sin(θ) * np.sin(φ))
              self.surface[surf]["z"] = np.ravel(self.surface[surf]["r"] * np.cos(θ))
              # add triangulation to dictionary
              self.surface[surf]["tri"] = Triangulation(np.ravel(θ), np.ravel(φ))