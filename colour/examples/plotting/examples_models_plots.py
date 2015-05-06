#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour models plotting examples.
"""

import numpy as np
from pprint import pprint

import colour
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Colour Models Plots')

message_box('Plotting "RGB" colourspaces in "CIE 1931 Chromaticity Diagram".')
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1931_chromaticity_diagram_plot(
    ['sRGB', 'ACES2065-1', 'Adobe RGB 1998', 'Pointer Gamut'])

print('\n')

message_box(('Plotting "RGB" colourspaces in '
             '"CIE 1960 UCS Chromaticity Diagram".'))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot(
    ['sRGB', 'ACES2065-1', 'Adobe RGB 1998', 'Pointer Gamut'])

print('\n')

message_box(('Plotting "RGB" colourspaces in '
             '"CIE 1976 UCS Chromaticity Diagram".'))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot(
    ['sRGB', 'ACES2065-1', 'Adobe RGB 1998', 'Pointer Gamut'])

print('\n')

RGB = np.random.random((32, 32, 3))

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1931 Chromaticity Diagram".')
RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot(
    RGB,
    colour.sRGB_COLOURSPACE,
    colourspaces=['ACEScg', 'sRGB', 'Pointer Gamut'])

print('\n')

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1960 UCS Chromaticity Diagram".')
RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot(
    RGB,
    colour.sRGB_COLOURSPACE,
    colourspaces=['ACEScg', 'sRGB', 'Pointer Gamut'])

print('\n')

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1976 UCS Chromaticity Diagram".')
RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot(
    RGB,
    colour.sRGB_COLOURSPACE,
    colourspaces=['ACEScg', 'sRGB', 'Pointer Gamut'])

print('\n')

message_box(('Plotting a single custom "RGB" colourspace in '
             '"CIE 1931 Chromaticity Diagram".'))
colour.RGB_COLOURSPACES['Awful RGB'] = colour.RGB_Colourspace(
    'Awful RGB',
    primaries=np.array([[0.1, 0.2],
                        [0.3, 0.15],
                        [0.05, 0.6]]),
    whitepoint=(1 / 3, 1 / 3))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1931_chromaticity_diagram_plot(['sRGB', 'Awful RGB'])

print('\n')

message_box('Plotting a single "RGB" colourspace transfer function.')
single_transfer_function_plot('sRGB')

print('\n')

message_box('Plotting multiple "RGB" colourspaces transfer functions.')
multi_transfer_function_plot(['sRGB', 'Rec. 709'])
