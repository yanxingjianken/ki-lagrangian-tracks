# coding: utf-8

from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from _cm import cmaps


def get_cm(name, levels=None):
    """ return a ListedColormap, a norm and the levels"""
    array = np.array(cmaps[name]['array'])
    if levels is None:
        levels = cmaps[name]['levels']
    cm = ListedColormap(array/255)
    norm = BoundaryNorm(levels, cm.N)
    return cm, norm, levels

# the dictionary to store the colormap
cmap_d = dict()

for cmap in cmaps:
    cmap_d[cmap] = get_cm(cmap)

locals().update(cmap_d)
