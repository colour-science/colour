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

colour_plotting_defaults()

message_box('Plotting "RGB" colourspaces in "CIE 1931 Chromaticity Diagram".')
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1931_chromaticity_diagram_plot(
    ['Rec. 709', 'ACEScg', 'S-Gamut', 'Pointer Gamut'])

print('\n')

message_box(('Plotting "RGB" colourspaces in '
             '"CIE 1960 UCS Chromaticity Diagram".'))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot(
    ['Rec. 709', 'ACEScg', 'S-Gamut', 'Pointer Gamut'])

print('\n')

message_box(('Plotting "RGB" colourspaces in '
             '"CIE 1976 UCS Chromaticity Diagram".'))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot(
    ['Rec. 709', 'ACEScg', 'S-Gamut', 'Pointer Gamut'])

print('\n')

RGB = np.random.random((32, 32, 3))

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1931 Chromaticity Diagram".')
RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot(
    RGB,
    'Rec. 709',
    colourspaces=['ACEScg', 'S-Gamut', 'Pointer Gamut'])

print('\n')

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1960 UCS Chromaticity Diagram".')
RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot(
    RGB,
    'Rec. 709',
    colourspaces=['ACEScg', 'S-Gamut', 'Pointer Gamut'])

print('\n')

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1976 UCS Chromaticity Diagram".')
RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot(
    RGB,
    'Rec. 709',
    colourspaces=['ACEScg', 'S-Gamut', 'Pointer Gamut'])

print('\n')

message_box(('Plotting a single custom "RGB" colourspace in '
             '"CIE 1931 Chromaticity Diagram".'))
colour.RGB_COLOURSPACES['Awful RGB'] = colour.RGB_Colourspace(
    'Awful RGB',
    primaries=np.array([[0.10, 0.20],
                        [0.30, 0.15],
                        [0.05, 0.60]]),
    whitepoint=np.array([1.0 / 3.0, 1.0 / 3.0]))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
RGB_colourspaces_CIE_1931_chromaticity_diagram_plot(['Rec. 709', 'Awful RGB'])

print('\n')

message_box(('Plotting a single "RGB" colourspace encoding colour component '
             'transfer function.'))
single_cctf_plot('Rec. 709')

print('\n')

message_box(('Plotting multiple "RGB" colourspaces encoding colour component '
             'transfer functions.'))
multi_cctf_plot(['Rec. 709', 'sRGB'])

message_box(('Plotting multiple "RGB" colourspaces decoding colour component '
             'transfer functions.'))
multi_cctf_plot(['ACES2065-1', 'ProPhoto RGB'], decoding_cctf=True)
