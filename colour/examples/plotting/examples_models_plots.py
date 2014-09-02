#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour models plotting examples.
"""

from numpy import array

from pprint import pprint
import colour
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Colour Models Plots')

message_box('Plotting "RGB" colourspaces in "CIE 1931 Chromaticity Diagram".')
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
colourspaces_CIE_1931_chromaticity_diagram_plot(
    ['sRGB', 'ACES RGB', 'Adobe RGB 1998'])

print('\n')

message_box(('Plotting a single custom "RGB" colourspace in '
             '"CIE 1931 Chromaticity Diagram".'))
colour.RGB_COLOURSPACES['Awful RGB'] = colour.RGB_Colourspace(
    'Awful RGB',
    primaries=array([[0.1, 0.2],
                     [0.3, 0.15],
                     [0.05, 0.6]]),
    whitepoint=(1 / 3, 1 / 3))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
colourspaces_CIE_1931_chromaticity_diagram_plot(['sRGB', 'Awful RGB'])

print('\n')

message_box('Plotting a single "RGB" colourspace transfer function.')
single_transfer_function_plot('sRGB')

print('\n')

message_box('Plotting multiple "RGB" colourspaces transfer functions.')
multi_transfer_function_plot(['sRGB', 'Rec. 709'])
