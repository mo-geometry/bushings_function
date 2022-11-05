import numpy as np
import sys
import random


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


def randomColormap(choice=True, colorcode=False, intense=False):
    if choice:
        x=['gist_earth', 'gist_yarg', 'gist_gray', 'ocean', 'cubehelix', 'binary',  'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'viridis', 'plasma', 'inferno', 'magma', 'Spectral', 'coolwarm', 'seismic']
    elif colorcode:
        x=['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', ]
    elif intense:
        x=['gist_ncar', 'gist_rainbow', 'gist_stern', 'nipy_spectral', 'hsv', 'bwr', 'jet', 'rainbow', 'brg', 'terrain', 'gnuplot', 'gnuplot2', 'CMRmap',]
    else:#pattern
        x=['Pastel1', 'Pastel2', 'Dark2', 'Accent', 'Set1', 'Set2', 'Set3', 'flag', 'Paired', 'prism', 'tab10', 'tab20', 'tab20b', 'tab20c',]
    return random.choice(x)