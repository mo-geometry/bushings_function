import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys


def initialize_figure(config, cigar, pancake):
    scale = np.min((1.0987 * np.max(pancake[0].r0), 1.0))
    # figure
    fig = plt.figure('surfaces of revolution from Bushing`s function', figsize=(12, 12))
    ax = plt.axes(projection='3d')
    # subplots
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('scaled function, first+second derivatives')
    ax4.set_xlabel('            θ')
    ax4.set_ylabel(' ')
    ###########
    # surface 1
    s1a = ax1.plot_trisurf(cigar[0].x, cigar[0].y, cigar[0].z, triangles=cigar[0].tri.triangles,
                     cmap=config["cmap1"], linewidths=0.1, alpha=0.88, edgecolor='Gray')
    ax1.view_init(elev=14, azim=30)
    s1b = ax1.text2D(0.05, 0.05, (' cmap: ' + config["cmap1"]
                            + '\n cigar harmonic oscillator '
                            + '\n   (η,λ,nF) = '
                            + '(%1.1f,' % config["η"]
                            + '%2d,' % config["λ"]
                            + '%2d) ' % config["nF"]),
                     bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 5},
                     transform=ax1.transAxes, ha="left")
    sub1 = [s1a, s1b, ax1]
    # surface 2
    s2a = ax2.plot_trisurf(pancake[0].x, pancake[0].y, pancake[0].z, triangles=pancake[0].tri.triangles,
                     cmap=config["cmap2"], linewidths=0.1, alpha=0.88, edgecolor='Gray')
    ax2.view_init(elev=14, azim=30)
    s2b = ax2.text2D(0.05, 0.05, (' cmap: ' + config["cmap2"]
                            + '\n pancake harmonic oscillator '
                            + '\n   (η,λ,nF) = '
                            + '(%1.1f,' % config["η"]
                            + '%2d,' % config["λ"]
                            + '%2d) ' % config["nF"]),
                     bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 5},
                     transform=ax2.transAxes, ha="left")
    sub2 = [s2a, s2b, ax2]
    # ~ fillers
    dθ0 = pancake[0].θ0[1] - pancake[0].θ0[0]
    dr0dθ_grad = np.gradient(pancake[0].r0, dθ0)
    d2r0dθ2_grad = np.gradient(dr0dθ_grad, dθ0)
    ###########
    # surface 3
    s3a, = ax3.plot(pancake[0].θ0, pancake[0].r0)
    #s3b = ax3.set_rmax(scale)
    # ax3.set_rticks([0.0, 0.25, 0.5, 0.75])  # Less radial ticks
    ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax3.set_xticklabels(['0', 'π/4', 'π/2', '3π/2', 'π', '5π/4', '6π/2', '7π/4'])
    ax3.grid(True)
    sub3 = [s3a,ax3]
    ###########
    # surface 4
    s4a, = ax4.plot(pancake[0].θ0, pancake[0].r0 - np.mean(pancake[0].r0), linewidth=2)
    s4b, = ax4.plot(pancake[0].θ0, pancake[0].dr0dθ)
    s4c, = ax4.plot(pancake[0].θ0, dr0dθ_grad, linestyle="dotted", linewidth=1.5)
    s4d, = ax4.plot(pancake[0].θ0, pancake[0].d2r0dθ2, alpha=0.11)
    s4e, = ax4.plot(pancake[0].θ0, d2r0dθ2_grad, linestyle="dotted", alpha=0.16)
    ax4.set_xlim(0, 2 * np.pi)
    ax4.set_ylim(-1, 1)
    ax4.set_xticks([0.0, (1 / 2) * np.pi, np.pi, (3 / 2) * np.pi, 2 * np.pi])  # Less radial ticks
    ax4.set_yticks([-1, -0.5, 0.0, 0.5, 1])  # Less radial ticks
    ax4.set_xticklabels([])  # Less radial ticks
    ax4.grid(True)
    sub4 = [s4a, s4b, s4c, s4d, s4e, ax4]
    return fig, sub1, sub2, sub3, sub4


def update_figure(idx, fig, cigar, pancake, sub1, sub2, sub3, sub4, config):
    scale = np.min((1.0987 * np.max(pancake[idx].r0), 1.0))
    # subplot 1
    sub1[0].remove()
    sub1[0] = sub1[-1].plot_trisurf(cigar[idx].x, cigar[idx].y, cigar[idx].z,
                          triangles=cigar[idx].tri.triangles,
                          cmap=config["cmap1"], linewidths=0.1, alpha=0.88, edgecolor='Gray'
                          )
    str1 = (' cmap: ' + config["cmap1"] + '\n cigar harmonic oscillator '
            + '\n   (η,λ,nF) = ' + '(%1.1f,' % cigar[idx].ηλnF[0] + '%2d,' % config["λ"] + '%2d) ' % config["nF"])
    sub1[1].set_text(str1)
    sub1[-1].set_xlim(-scale, scale)
    sub1[-1].set_ylim(-scale, scale)
    sub1[-1].set_zlim(-scale, scale)
    # subplot 2
    sub2[0].remove()
    sub2[0] = sub2[-1].plot_trisurf(pancake[idx].x, pancake[idx].y, pancake[idx].z,
                          triangles=pancake[idx].tri.triangles,
                          cmap=config["cmap2"], linewidths=0.1, alpha=0.88, edgecolor='Gray'
                          )
    str2 = (' cmap: ' + config["cmap2"] + '\n pancake harmonic oscillator '
            + '\n   (η,λ,nF) = ' + '(%1.1f,' % pancake[idx].ηλnF[0] + '%2d,' % config["λ"] + '%2d) ' % config["nF"])
    sub2[1].set_text(str2)
    sub2[-1].set_xlim(-scale, scale)
    sub2[-1].set_ylim(-scale, scale)
    sub2[-1].set_zlim(-scale, scale)
    # subplot 3
    sub3[0].set_ydata(pancake[idx].r0)
    sub3[-1].set_rmax(scale)    # ax is last element of the list
    # ~ fillers
    dθ0 = pancake[idx].θ0[1] - pancake[idx].θ0[0]
    dr0dθ_grad = np.gradient(pancake[idx].r0, dθ0)
    d2r0dθ2_grad = np.gradient(dr0dθ_grad, dθ0)
    # subplot 4
    sub4[0].set_ydata(pancake[idx].r0 - np.mean(pancake[idx].r0))
    sub4[1].set_ydata(pancake[idx].dr0dθ)
    sub4[2].set_ydata(dr0dθ_grad)
    sub4[3].set_ydata(pancake[idx].d2r0dθ2)
    sub4[4].set_ydata(d2r0dθ2_grad)
    # update progress bar
    drawProgressBar(idx / config["frames"], barLen=50)
    return fig, sub1, sub2, sub3, sub4



# progress bar for data loading
def drawProgressBar(percent, barLen=20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%% " % (progress, percent * 100))
    sys.stdout.flush()


def randomColormap(choice=False, colorcode=False, intense=False):
    if choice:
        x=['gist_earth', 'gist_yarg', 'gist_gray', 'ocean', 'cubehelix', 'binary',  'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'viridis', 'plasma', 'inferno', 'magma', 'Spectral', 'coolwarm', 'seismic']
    elif colorcode:
        x=['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', ]
    elif intense:
        x=['gist_ncar', 'gist_rainbow', 'gist_stern', 'nipy_spectral', 'hsv', 'bwr', 'jet', 'rainbow', 'brg', 'terrain', 'gnuplot', 'gnuplot2', 'CMRmap',]
    else:#pattern
        x=['Pastel1', 'Pastel2', 'Dark2', 'Accent', 'Set1', 'Set2', 'Set3', 'flag', 'Paired', 'prism', 'tab10', 'tab20', 'tab20b', 'tab20c',]
    return random.choice(x)