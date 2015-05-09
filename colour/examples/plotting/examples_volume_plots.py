#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour models volume and gamut plotting examples.
"""

from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

message_box('Colour Models Volume and Gamut Plots')

message_box(('Plotting "Rec. 709" RGB colourspace volume in "CIE xyY" '
             'colourspace.'))
RGB_colourspaces_gamut_plot(('Rec. 709',), reference_colourspace='CIE xyY')

print('\n')

message_box(('Comparing "Rec. 709" and "ACEScg" RGB colourspaces volume '
             'in "CIE Lab" colourspace.'))
RGB_colourspaces_gamut_plot(('Rec. 709', 'ACEScg'),
                            reference_colourspace='CIE Lab',
                            style={
                                'face_colours': (None, (0.25, 0.25, 0.25)),
                                'edge_colours': (None, (0.25, 0.25, 0.25)),
                                'edge_alpha': (1.0, 0.1),
                                'face_alpha': (1.0, 0.0)})
