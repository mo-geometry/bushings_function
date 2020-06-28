######################################################
#   SURFACES OF REVOLUTION FROM BUSHING'S FUNCTION   #
######################################################
#
# Bushing's function is derived from the coherent
# states of the anisotropic harmonic oscillator.
#
# 1)
# Inhibition of spontaneous emission in Fermi gases
# Busch et al, Euro-phys. Letters 44, 1 (1998).
# DOI: https://doi.org/10.1209/epl/i1998-00426-2
# 2)
# Spontaneous Emission in ultra-cold spin-polarised
# anisotropic Fermi Seas
# O'Sullivan & Busch, Phys. Rev. A 79, 033602 (2009).
# DOI: https://doi.org/10.1103/PhysRevA.79.033602
# arXiv: 0810.0231
# 3)
# Spatial and energetic mode dynamics of cold
# atomic systems, O'Sullivan.
# - see chapter 2.3 for a full derivation of
# equation (2.112)
# URL: https://cora.ucc.ie/handle/10468/963

# MEDITATIONS ON GEOMETRY ##############################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import sys
from plotFunctions import *
from bushingsFunction import *

# PREAMBLE #############################################################################################################

config = {"cmap1": randomColormap(choice=True, colorcode=False, intense=False),
          "cmap2": randomColormap(choice=False, colorcode=True, intense=False),
          "η": 5.0,                     # spread: float point value
          "λ": 11,                      # anisotropy: positive integer excluding 0
          "nF": 23,                     # occupancy: positive integer including 0
          "frames": 2 ** 6,             # number of frames for image (approx 96 is max depending on memory)
          "animation": False            # if False display a single image
          }

# MAIN #################################################################################################################

# load the surfaces
cigar = [Surface(config, cigar=True)]
pancake = [Surface(config, pancake=True)]

# append data for the animation
if config["animation"]:
    for idx in range(config["frames"]):
        θidx = idx/config["frames"] * 2 * np.pi
        config["η"] = 5 + 4 * np.sin(θidx) + 3 * np.sin(θidx/2) ** 2
        cigar.append(Surface(config, cigar=True))
        pancake.append(Surface(config, pancake=True))
        drawProgressBar(idx/config["frames"], barLen=50)
    sys.stdout.write('\r')   # remove progress bar

# initialize figure
fig, sub1, sub2, sub3, sub4 = initialize_figure(config, cigar, pancake)

# animate the figure
if config["animation"]:
    # animate the figure
    anim = animation.FuncAnimation(fig, update_figure,
                                   frames=config["frames"], repeat=False, interval=1,
                                   fargs=(fig, cigar, pancake, sub1, sub2, sub3, sub4, config))
    # anim.save('bushings.gif', dpi=80, writer='imagemagick')
    # anim.save("frames0/bushings.png", writer="imagemagick")

# FIN ##################################################################################################################

plt.show()
