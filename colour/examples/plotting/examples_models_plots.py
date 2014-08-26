#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour models plotting examples.
"""

from numpy import array

import colour
from colour.plotting import *

# Plotting colourspaces in *CIE 1931 Chromaticity Diagram*.
print(sorted(colour.RGB_COLOURSPACES.keys()))
colourspaces_CIE_1931_chromaticity_diagram_plot(
    ['sRGB', 'ACES RGB', 'Adobe RGB 1998'])

# Plotting a single custom colourspace in *CIE 1931 Chromaticity Diagram*.
colour.RGB_COLOURSPACES['Awful RGB'] = colour.RGB_Colourspace(
    'Awful RGB',
    primaries=array([[0.1, 0.2],
                     [0.3, 0.15],
                     [0.05, 0.6]]),
    whitepoint=(1 / 3, 1 / 3))

print(sorted(colour.RGB_COLOURSPACES.keys()))
colourspaces_CIE_1931_chromaticity_diagram_plot(['sRGB', 'Awful RGB'])

# Plotting a single colourspace transfer function.
single_transfer_function_plot('sRGB')

# Plotting multiple colourspaces transfer functions.
multi_transfer_function_plot(['sRGB', 'Rec. 709'])
