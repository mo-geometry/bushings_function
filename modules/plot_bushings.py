import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
import glob
from datetime import datetime
from modules.objects import *

# OBJECTS ##############################################################################################################

class FIGURE:
  def __init__(self, bushings):
      self.bushings, self.args = bushings, None
      self.cmap_cigar = randomColormap()
      self.cmap_pancake = randomColormap()
      self.fig, self.subplots = [], []


  def update(self, bushings):
      sub1, sub2, sub3, sub4 = self.subplots[0], self.subplots[1], self.subplots[2], self.subplots[3]
      # EXTRACT
      η, λ, nF = bushings.settings["η"], bushings.settings["λ"], bushings.settings["nF"]
      # Textboxes
      textbox_cigar = 'cigar harmonic oscillator ' + '\n   (η,λ,nF) = (%1.1f,%2d,%2d)' % (η, λ, nF) + \
                      '\n cmap: ' + self.cmap_cigar
      textbox_pancake = 'pancake harmonic oscillator ' + '\n   (η,λ,nF) = (%1.1f,%2d,%2d)' % (η, λ, nF) + \
                        '\n cmap: ' + self.cmap_pancake
      if self.bushings.type[0] == "sphere":
          textbox_cigar = "sphere" + '\n cmap: ' + self.cmap_cigar
          textbox_pancake = "sphere" + '\n cmap: ' + self.cmap_pancake
          surf = ["sphere", "sphere"]
      else:
          surf = list(bushings.surface.keys())
      # profile in the 2D plane
      θ0, r0 = bushings.plane[surf[0]]["θ"], bushings.plane[surf[0]]["r"]
      # gradients - numerical
      dθ0 = bushings.plane[surf[0]]["θ"][1] - bushings.plane[surf[0]]["θ"][0]
      dr0dθ_grad = np.gradient(bushings.plane[surf[0]]["r"], dθ0)
      d2r0dθ2_grad = np.gradient(dr0dθ_grad, dθ0)
      # gradients - calculated
      dr0dθ = bushings.plane[surf[0]]["dr"]
      d2r0dθ2 = bushings.plane[surf[0]]["d2r"]
      # surfaces
      cigar_x = bushings.surface[surf[0]]["x"]
      cigar_y = bushings.surface[surf[0]]["y"]
      cigar_z = bushings.surface[surf[0]]["z"]
      cigar_tri = bushings.surface[surf[0]]["tri"].triangles
      pancake_x = bushings.surface[surf[1]]["x"]
      pancake_y = bushings.surface[surf[1]]["y"]
      pancake_z = bushings.surface[surf[1]]["z"]
      pancake_tri = bushings.surface[surf[1]]["tri"].triangles
      # scale limits
      scale = np.min((1.0987 * np.max(r0), 1.0))
      # subplot 1
      sub1[0].remove()
      sub1[0] = sub1[-1].plot_trisurf(cigar_x, cigar_y, cigar_z, triangles=cigar_tri, cmap=self.cmap_cigar,
                                      linewidths=0.1, alpha=0.88, edgecolor='Gray')
      sub1[1].set_text(textbox_cigar)
      sub1[-1].set_xlim(-scale, scale)
      sub1[-1].set_ylim(-scale, scale)
      sub1[-1].set_zlim(-scale, scale)
      # subplot 2
      sub2[0].remove()
      sub2[0] = sub2[-1].plot_trisurf(pancake_x, pancake_y, pancake_z, triangles=pancake_tri, cmap=self.cmap_pancake,
                                      linewidths=0.1, alpha=0.88, edgecolor='Gray')
      sub2[1].set_text(textbox_pancake)
      sub2[-1].set_xlim(-scale, scale)
      sub2[-1].set_ylim(-scale, scale)
      sub2[-1].set_zlim(-scale, scale)
      # subplot 3
      sub3[0].set_ydata(r0)
      sub3[-1].set_rmax(scale)  # ax is last element of the list
      # subplot 4
      sub4[0].set_ydata(r0 - np.mean(r0))
      sub4[1].set_ydata(dr0dθ)
      sub4[2].set_ydata(dr0dθ_grad)
      sub4[3].set_ydata(d2r0dθ2)
      sub4[4].set_ydata(d2r0dθ2_grad)
      # show figure
      if self.args.animation:
          plt.pause(1e-6)
      if self.args.save_frames_for_avi:
          root = os.path.join(os.getcwd(), self.args.figures_folder)
          plt.savefig(os.path.join(root, "figure " + str(datetime.now()) + ".png"), bbox_inches='tight', pad_inches=0.1)



  # BUSHINGS FUNCTION: SURFACES OF REVOLUTION
  def plot(self, args):
      self.args = args
      # EXTRACT
      η, λ, nF = self.bushings.settings["η"], self.bushings.settings["λ"], self.bushings.settings["nF"]
      # Textboxes
      textbox_cigar = 'cigar harmonic oscillator ' + '\n   (η,λ,nF) = (%1.1f,%2d,%2d)' % (η, λ, nF) +\
                      '\n cmap: ' + self.cmap_cigar
      textbox_pancake = 'pancake harmonic oscillator ' + '\n   (η,λ,nF) = (%1.1f,%2d,%2d)' % (η, λ, nF) +\
                        '\n cmap: ' + self.cmap_pancake
      if self.bushings.type[0] == "sphere":
          textbox_cigar = "sphere" + '\n cmap: ' + self.cmap_cigar
          textbox_pancake = "sphere" + '\n cmap: ' + self.cmap_pancake
          surf = ["sphere", "sphere"]
      else:
          surf = list(self.bushings.surface.keys())
      # profile in the 2D plane
      θ0, r0 = self.bushings.plane[surf[0]]["θ"], self.bushings.plane[surf[0]]["r"]
      # gradients - numerical
      dθ0 = self.bushings.plane[surf[0]]["θ"][1] - self.bushings.plane[surf[0]]["θ"][0]
      dr0dθ_grad = np.gradient(self.bushings.plane[surf[0]]["r"], dθ0)
      d2r0dθ2_grad = np.gradient(dr0dθ_grad, dθ0)
      # gradients - calculated
      dr0dθ = self.bushings.plane[surf[0]]["dr"]
      d2r0dθ2 = self.bushings.plane[surf[0]]["d2r"]
      # surfaces
      cigar_x = self.bushings.surface[surf[0]]["x"]
      cigar_y = self.bushings.surface[surf[0]]["y"]
      cigar_z = self.bushings.surface[surf[0]]["z"]
      cigar_tri = self.bushings.surface[surf[0]]["tri"].triangles
      pancake_x = self.bushings.surface[surf[1]]["x"]
      pancake_y = self.bushings.surface[surf[1]]["y"]
      pancake_z = self.bushings.surface[surf[1]]["z"]
      pancake_tri = self.bushings.surface[surf[1]]["tri"].triangles
      # figure
      plt.ion()
      fig = plt.figure('surfaces of revolution from Bushing`s function',
                       figsize=(args.figure_width_height[0], args.figure_width_height[1]))
      fig.suptitle("\n" + r"$\bf{Bushings\ Function:}$ Surfaces of Revolution", fontsize=14)
      ax1 = plt.subplot(2, 2, 1, projection='3d')
      ax2 = plt.subplot(2, 2, 2, projection='3d')
      ax3 = plt.subplot(2, 2, 3, projection='polar')
      ax4 = plt.subplot(2, 2, 4)
      ax4.set_title('scaled function, first+second derivatives')
      ax4.set_xlabel('            θ')
      ax4.set_ylabel(' ')
      ###########
      # surface 1
      s1a = ax1.plot_trisurf(cigar_x, cigar_y, cigar_z, triangles=cigar_tri,
                             cmap=self.cmap_cigar, linewidths=0.1, alpha=0.88, edgecolor='Gray')
      ax1.view_init(elev=14, azim=30)
      s1b = ax1.text2D(0.05, 0.05, textbox_cigar, bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 5},
                       transform=ax1.transAxes, ha="left")
      sub1 = [s1a, s1b, ax1]
      # surface 2
      s2a = ax2.plot_trisurf(pancake_x, pancake_y, pancake_z, triangles=pancake_tri,
                             cmap=self.cmap_pancake, linewidths=0.1, alpha=0.88, edgecolor='Gray')
      ax2.view_init(elev=14, azim=30)
      s2b = ax2.text2D(0.05, 0.05, textbox_pancake, bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 5},
                       transform=ax2.transAxes, ha="left")
      sub2 = [s2a, s2b, ax2]
      ###########
      # surface 3
      s3a, = ax3.plot(θ0, r0)
      # s3b = ax3.set_rmax(scale)
      # ax3.set_rticks([0.0, 0.25, 0.5, 0.75])  # Less radial ticks
      ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
      ax3.set_xticklabels(['0', 'π/4', 'π/2', '3π/2', 'π', '5π/4', '6π/2', '7π/4'])
      ax3.grid(True)
      sub3 = [s3a, ax3]
      ###########
      # surface 4
      s4a, = ax4.plot(θ0, r0 - np.mean(r0), linewidth=2)
      s4b, = ax4.plot(θ0, dr0dθ)
      s4c, = ax4.plot(θ0, dr0dθ_grad, linestyle="dotted", linewidth=1.5)
      s4d, = ax4.plot(θ0, d2r0dθ2, alpha=0.11)
      s4e, = ax4.plot(θ0, d2r0dθ2_grad, linestyle="dotted", alpha=0.16)
      ax4.set_xlim(0, 2 * np.pi)
      ax4.set_ylim(-1, 1)
      ax4.set_xticks([0.0, (1 / 2) * np.pi, np.pi, (3 / 2) * np.pi, 2 * np.pi])  # Less radial ticks
      ax4.set_yticks([-1, -0.5, 0.0, 0.5, 1])  # Less radial ticks
      ax4.set_xticklabels([])  # Less radial ticks
      ax4.grid(True)
      sub4 = [s4a, s4b, s4c, s4d, s4e, ax4]
      self.fig = fig
      self.subplots = [sub1, sub2, sub3, sub4]
      if args.save_frames_for_avi:
          root = os.path.join(os.getcwd(), args.figures_folder)
          if not os.path.exists(root):
              os.mkdir(root)
          else:
              shutil.rmtree(root)
              os.mkdir(root)
          # save frame
          if args.animation is False:
              plt.savefig(os.path.join(root, "figure " + str(datetime.now()) + ".png"),
                          bbox_inches='tight', pad_inches=0.1)

