######################################################
#   SURFACES OF REVOLUTION FROM BUSHING'S FUNCTION   #
######################################################
#
# Bushing's function is derived from the coherent
# states of the anisotropic harmonic oscillator.
#
# 1)
# Inhibition of spontaneous emission in Fermi gases
# Busch et al., Euro-phys. Letters 44, 1 (1998).
# DOI: https://doi.org/10.1209/epl/i1998-00426-2
# 2)
# Spontaneous Emission in ultra-cold spin-polarised
# anisotropic Fermi Seas
# O'Sullivan & Busch, Phys. Rev. A 79, 033602 (2009).
# DOI: https://doi.org/10.1103/PhysRevA.79.033602
# arXiv: 0810.0231
# 3)
# Spatial and energetic mode dynamics of cold
# atomic systems, PhD Thesis [2012] Brian O'Sullivan.
# - see chapter 2.3 for a full derivation of
# equation (2.112)
# URL: https://cora.ucc.ie/handle/10468/963

# MEDITATIONS ON GEOMETRY ##############################################################################################
import cv2
import sys
import argparse
from modules.bushings_function import *
from modules.plot_bushings import *

# ARGUMENTS ############################################################################################################
parser = argparse.ArgumentParser(description='Bushings Function')
# surface options & intial state
parser.add_argument('--frames', default=2 ** 6)
parser.add_argument('--fps', default=8)
parser.add_argument('--save_frames_for_avi', default=True)
parser.add_argument('--delete_avi_figures', default=True)
parser.add_argument('--figure_width_height', default=(9, 8))
parser.add_argument('--figures_folder', default="figures")
args = parser.parse_args()

# MAIN #################################################################################################################

# initialize figure
fig = FIGURE(BUSHINGS())
fig.plot(args)
plt.show()

# animation
if args.save_frames_for_avi:
    A = np.linspace(0, 2 * np.pi, args.frames)
    η = 5 + 4 * np.sin(A) + 3 * np.sin(A / 2) ** 2
    sys.stdout.write("Generating images:\n")
    for idx in range(args.frames):
        # update figure
        fig.update(BUSHINGS(η=η[idx]))
        # update progress bar
        drawProgressBar((idx + 1) / args.frames, barLen=50)
    # remove progress bar
    sys.stdout.write('\r')
    sys.stdout.write(" *** done *** \n")
else:
    # delete __pycache__ folder
    shutil.rmtree(os.path.join(os.getcwd(), "modules", "__pycache__"))
    sys.stdout.write(" *** done *** \n")
    raise SystemExit

# VID ##################################################################################################################

# Write video
sys.stdout.write("Writing video:\n")

# load images
image_folder = os.path.join(os.getcwd(), args.figures_folder)
images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

# setting the frame (height, width) according to the (height, width) of first image
height, width, layers = cv2.imread(os.path.join(image_folder, images[0])).shape
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter("bushings.avi", fourcc, args.fps, (width, height))

# Appending the images to the video one by one
for image, idx in zip(images, range(len(images))):
    video.write(cv2.imread(os.path.join(image_folder, image)))
    # update progress bar
    drawProgressBar((idx + 1) / len(images), barLen=50)
sys.stdout.write('\r')      # remove progress bar

# Deallocating memories taken for window creation
cv2.destroyAllWindows()
video.release()  # releasing the video generated
# delete image folder
if args.delete_avi_figures:
    shutil.rmtree(image_folder)
# delete __pycache__ folder
shutil.rmtree(os.path.join(os.getcwd(), "modules", "__pycache__"))
sys.stdout.write(" *** done *** \n")

# FIN ##################################################################################################################
