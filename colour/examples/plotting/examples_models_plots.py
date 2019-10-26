# -*- coding: utf-8 -*-
"""
Showcases colour models plotting examples.
"""

import numpy as np
from pprint import pprint

import colour
from colour.plotting import (
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS, colour_style,
    plot_multi_cctfs, plot_single_cctf)
from colour.utilities import message_box

message_box('Colour Models Plots')

colour_style()

message_box('Plotting "RGB" colourspaces in "CIE 1931 Chromaticity Diagram".')
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
    ['ITU-R BT.709', 'ACEScg', 'S-Gamut'], show_pointer_gamut=True)

print('\n')

message_box(('Plotting "RGB" colourspaces in '
             '"CIE 1960 UCS Chromaticity Diagram".'))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS(
    ['ITU-R BT.709', 'ACEScg', 'S-Gamut'], show_pointer_gamut=True)

print('\n')

message_box(('Plotting "RGB" colourspaces in '
             '"CIE 1976 UCS Chromaticity Diagram".'))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(
    ['ITU-R BT.709', 'ACEScg', 'S-Gamut'], show_pointer_gamut=True)

print('\n')

RGB = np.random.random((32, 32, 3))

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1931 Chromaticity Diagram".')
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
    RGB,
    'ITU-R BT.709',
    colourspaces=['ACEScg', 'S-Gamut'],
    show_pointer_gamut=True)

print('\n')

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1960 UCS Chromaticity Diagram".')
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(
    RGB,
    'ITU-R BT.709',
    colourspaces=['ACEScg', 'S-Gamut'],
    show_pointer_gamut=True)

print('\n')

message_box('Plotting "RGB" chromaticity coordinates in '
            '"CIE 1976 UCS Chromaticity Diagram".')
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
    RGB,
    'ITU-R BT.709',
    colourspaces=['ACEScg', 'S-Gamut'],
    show_pointer_gamut=True)

print('\n')

message_box(('Plotting a single custom "RGB" colourspace in '
             '"CIE 1931 Chromaticity Diagram".'))
colour.RGB_COLOURSPACES['Awful RGB'] = colour.RGB_Colourspace(
    'Awful RGB',
    primaries=np.array([
        [0.10, 0.20],
        [0.30, 0.15],
        [0.05, 0.60],
    ]),
    whitepoint=np.array([1.0 / 3.0, 1.0 / 3.0]))
pprint(sorted(colour.RGB_COLOURSPACES.keys()))
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
    ['ITU-R BT.709', 'Awful RGB'])

print('\n')

message_box(('Plotting a single "RGB" colourspace encoding colour component '
             'transfer function.'))
plot_single_cctf('ITU-R BT.709')

print('\n')

message_box(('Plotting multiple "RGB" colourspaces encoding colour component '
             'transfer functions.'))
plot_multi_cctfs(['ITU-R BT.709', 'sRGB'])

message_box(('Plotting multiple "RGB" colourspaces decoding colour component '
             'transfer functions.'))
plot_multi_cctfs(['ACES2065-1', 'ProPhoto RGB'], cctf_decoding=True)
