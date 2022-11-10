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


  def return_textbox(self, η, λ, nF):
      # Textboxes
      textbox_cigar = 'cigar harmonic oscillator ' + '\n   (η,λ,nF) = (%1.1f,%2d,%2d)' % (η, λ, nF) + \
                      '\n cmap: ' + self.cmap_cigar
      textbox_pancake = 'pancake harmonic oscillator ' + '\n   (η,λ,nF) = (%1.1f,%2d,%2d)' % (η, λ, nF) + \
                        '\n cmap: ' + self.cmap_pancake
      return textbox_cigar, textbox_pancake

  @staticmethod
  def return_gradients(dictionary):
      dθ0 = dictionary["θ"][1] - dictionary["θ"][0]
      dr0dθ_grad = np.gradient(dictionary["r"], dθ0)
      d2r0dθ2_grad = np.gradient(dr0dθ_grad, dθ0)
      return dr0dθ_grad, d2r0dθ2_grad

  @staticmethod
  def return_surface(dictionary):
      x, y, z = dictionary["x"], dictionary["y"], dictionary["z"]
      tri = dictionary["tri"].triangles
      return [x, y, z], tri

  def update(self, bushings):
      sub1, sub2, sub3, sub4 = self.subplots[0], self.subplots[1], self.subplots[2], self.subplots[3]
      # Textboxes
      η, λ, nF = bushings.settings["η"], bushings.settings["λ"], bushings.settings["nF"]
      textbox_cigar, textbox_pancake = self.return_textbox(η, λ, nF)
      # profile in the 2D plane [cigar]
      pancake_2D_plane_dict = bushings.plane["pancake"]
      θ0, r0 = pancake_2D_plane_dict["θ"], pancake_2D_plane_dict["r"]
      dr0dθ_grad, d2r0dθ2_grad = self.return_gradients(pancake_2D_plane_dict)  # gradients - numerical
      dr0dθ, d2r0dθ2 = pancake_2D_plane_dict["dr"], pancake_2D_plane_dict["d2r"]  # gradients - calculated
      # surfaces
      pancake_xyz, pancake_tri = self.return_surface(bushings.surface["pancake"])
      cigar_xyz, cigar_tri = self.return_surface(bushings.surface["cigar"])
      # scale limits
      scale = np.min((1.0987 * np.max(r0), 1.0))
      # PLOTTING
      # subplot 1
      sub1[0].remove()
      sub1[0] = sub1[-1].plot_trisurf(pancake_xyz[0], pancake_xyz[1], pancake_xyz[2], triangles=pancake_tri,
                                      cmap=self.cmap_pancake, linewidths=0.1, alpha=0.88, edgecolor='Gray')
      sub1[1].set_text(textbox_pancake)
      sub1[-1].set_xlim(-scale, scale)
      sub1[-1].set_ylim(-scale, scale)
      sub1[-1].set_zlim(-scale, scale)
      # subplot 2
      sub2[0].remove()
      sub2[0] = sub2[-1].plot_trisurf(cigar_xyz[0], cigar_xyz[1], cigar_xyz[2], triangles=cigar_tri,
                                      cmap=self.cmap_cigar, linewidths=0.1, alpha=0.88, edgecolor='Gray')
      sub2[1].set_text(textbox_cigar)
      sub2[-1].set_xlim(-scale, scale)
      sub2[-1].set_ylim(-scale, scale)
      sub2[-1].set_zlim(-scale, scale)
      # subplot 3
      sub3[0].set_ydata(r0)
      sub3[1].set_rmax(scale)
      # subplot 4
      sub4[0].set_ydata(r0 - np.mean(r0))
      sub4[1].set_ydata(dr0dθ)
      sub4[2].set_ydata(dr0dθ_grad)
      sub4[3].set_ydata(d2r0dθ2)
      sub4[4].set_ydata(d2r0dθ2_grad)
      # show figure
      if self.args.save_frames_for_avi:
          plt.pause(1e-6)
          root = os.path.join(os.getcwd(), self.args.figures_folder)
          plt.savefig(os.path.join(root, "figure " + str(datetime.now()) + ".png"), bbox_inches='tight', pad_inches=0.1)

  # BUSHINGS FUNCTION: SURFACES OF REVOLUTION
  def plot(self, args):
      self.args = args
      # Textboxes
      η, λ, nF = self.bushings.settings["η"], self.bushings.settings["λ"], self.bushings.settings["nF"]
      textbox_cigar, textbox_pancake = self.return_textbox(η, λ, nF)
      # profile in the 2D plane [pancake]
      pancake_2D_plane_dict = self.bushings.plane["pancake"]
      θ0, r0 = pancake_2D_plane_dict["θ"], pancake_2D_plane_dict["r"]
      dr0dθ_grad, d2r0dθ2_grad = self.return_gradients(pancake_2D_plane_dict) # gradients - numerical
      dr0dθ, d2r0dθ2 = pancake_2D_plane_dict["dr"], pancake_2D_plane_dict["d2r"]  # gradients - calculated
      # surfaces
      pancake_xyz, pancake_tri = self.return_surface(self.bushings.surface["pancake"])
      cigar_xyz, cigar_tri = self.return_surface(self.bushings.surface["cigar"])
      # figure - initialize
      plt.ion()
      self.fig = plt.figure('surfaces of revolution from Bushing`s function',
                            figsize=(args.figure_width_height[0], args.figure_width_height[1]))
      self.fig.suptitle("\n" + r"$\bf{Bushings\ Function:}$ Surfaces of Revolution", fontsize=14)
      ax1 = plt.subplot(2, 2, 1, projection='3d')
      ax2 = plt.subplot(2, 2, 2, projection='3d')
      ax3 = plt.subplot(2, 2, 3, projection='polar')
      ax4 = plt.subplot(2, 2, 4)
      ###########
      # surface 1
      s1a = ax1.plot_trisurf(pancake_xyz[0], pancake_xyz[1], pancake_xyz[2], triangles=pancake_tri,
                             cmap=self.cmap_pancake, linewidths=0.1, alpha=0.88, edgecolor='Gray')
      s1b = ax1.text2D(0.05, 0.05, textbox_pancake, bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 5},
                      transform=ax1.transAxes, ha="left")
      ax1.view_init(elev=14, azim=30)
      sub1 = [s1a, s1b, ax1]    # save to list for plot update
      # surface 2
      s2a = ax2.plot_trisurf(cigar_xyz[0], cigar_xyz[1], cigar_xyz[2], triangles=cigar_tri,
                            cmap=self.cmap_cigar, linewidths=0.1, alpha=0.88, edgecolor='Gray')
      s2b = ax2.text2D(0.05, 0.05, textbox_cigar, bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 5},
                      transform=ax2.transAxes, ha="left")
      ax2.view_init(elev=14, azim=30)
      sub2 = [s2a, s2b, ax2]    # save to list for plot update
      ###########
      # surface 3
      s3, = ax3.plot(θ0, r0)
      sub3 = [s3, ax3]         # save to list for plot update
      ###########
      # surface 4
      s4a, = ax4.plot(θ0, r0 - np.mean(r0), linewidth=2, label='Surface')
      s4b, = ax4.plot(θ0, dr0dθ, label='1st derivative')
      s4c, = ax4.plot(θ0, dr0dθ_grad, linestyle="dotted", linewidth=1.5)
      s4d, = ax4.plot(θ0, d2r0dθ2, alpha=0.11, label='2nd derivative')
      s4e, = ax4.plot(θ0, d2r0dθ2_grad, linestyle="dotted", alpha=0.16)
      sub4 = [s4a, s4b, s4c, s4d, s4e, ax4]     # save to list for plot update
      # set labels and limits
      # SUBPLOT 3
      ax3.set_theta_zero_location("N")  # Move radial labels away from plotted line
      ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
      ax3.set_xticklabels(['0', 'π/4', 'π/2', '3π/2', 'π', '5π/4', '6π/2', '7π/4'])
      ax3.grid(True)
      # SUBPLOT 4
      ax4.set_title('surface derivatives')
      ax4.set_xlabel('            θ')
      ax4.set_ylabel(' ')
      ax4.set_xlim(0, 2 * np.pi)
      ax4.set_ylim(-1, 1)
      ax4.set_yticks([-1, -0.5, 0.0, 0.5, 1])  # Less radial ticks
      ax4.set_xticks([0.0, (1 / 2) * np.pi, np.pi, (3 / 2) * np.pi, 2 * np.pi])  # Less radial ticks
      ax4.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])  # Less radial ticks
      ax4.grid(True)
      ax4.legend(loc='lower right')
      # assign # save to list for plot update
      self.subplots = [sub1, sub2, sub3, sub4]
      # save first figure
      root = os.path.join(os.getcwd(), args.figures_folder)
      if not os.path.exists(root):
          os.mkdir(root)
      else:
          shutil.rmtree(root)
          os.mkdir(root)
      # save frame
      plt.savefig(os.path.join(root, "figure " + str(datetime.now()) + ".png"),
                  bbox_inches='tight', pad_inches=0.1)
      if self.bushings.type[0] == "sphere":
          # delete __pycache__ folder
          shutil.rmtree(os.path.join(os.getcwd(), "modules", "__pycache__"))
          sys.stdout.write(" *** animation discontinued for sphere-type surface *** \n")
          sys.stdout.write(" *** done *** \n")
          raise SystemExit

