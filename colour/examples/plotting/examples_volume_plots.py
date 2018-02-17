# -*- coding: utf-8 -*-
"""
Showcases colour models volume and gamut plotting examples.
"""

import numpy as np

from colour.plotting import (RGB_colourspaces_gamuts_plot, RGB_scatter_plot,
                             colour_plotting_defaults)
from colour.utilities import message_box

message_box('Colour Models Volume and Gamut Plots')

colour_plotting_defaults()

message_box(('Plotting "ITU-R BT.709" RGB colourspace volume in "CIE xyY" '
             'colourspace.'))
RGB_colourspaces_gamuts_plot(
    ('ITU-R BT.709', ), reference_colourspace='CIE xyY')

print('\n')

message_box(('Comparing "ITU-R BT.709" and "ACEScg" RGB colourspaces volume '
             'in "CIE L*a*b*" colourspace.'))
RGB_colourspaces_gamuts_plot(
    ('ITU-R BT.709', 'ACEScg'),
    reference_colourspace='CIE Lab',
    style={
        'face_colours': (None, (0.25, 0.25, 0.25)),
        'edge_colours': (None, (0.25, 0.25, 0.25)),
        'edge_alpha': (1.0, 0.1),
        'face_alpha': (1.0, 0.0)
    })

print('\n')

message_box(('Plotting "ACEScg" colourspaces values in "CIE L*a*b*" '
             'colourspace.'))

RGB = np.random.random((32, 32, 3))

RGB_scatter_plot(
    RGB,
    'ACEScg',
    reference_colourspace='CIE Lab',
    colourspaces=('ACEScg', 'ITU-R BT.709'),
    face_colours=((0.25, 0.25, 0.25), None),
    edge_colours=((0.25, 0.25, 0.25), None),
    edge_alpha=(0.1, 0.5),
    face_alpha=(0.1, 0.5),
    grid_face_colours=(0.1, 0.1, 0.1),
    grid_edge_colours=(0.1, 0.1, 0.1),
    grid_edge_alpha=0.5,
    grid_face_alpha=0.1)
