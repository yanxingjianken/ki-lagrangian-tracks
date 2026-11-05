# coding: utf-8

"""
File to store dictionary of LinearSegmentedColormaps
and a dictionary of these dictionaries.

Based on the matplotlib.cm module
"""

_pv_data = [
    [127, 151, 255],
    [142, 178, 255],
    [181, 201, 255],
    [215, 227, 237],
    [246, 221, 160],
    [239, 192, 130],
    [242, 131, 67],
    [206, 4, 4],
    [254, 121, 19],
    [255, 209, 33],
    [255, 249, 19],
    [255, 255, 192]
]
_pv_levels = [-1.0, 0, 0.2, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 8, 9]


cmaps = {
    'PV': {'array': _pv_data,
           'levels': _pv_levels},
}
