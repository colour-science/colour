#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour rendition charts computations.
"""

from __future__ import division, unicode_literals

import numpy as np
from pprint import pprint

import colour
from colour.utilities.verbose import message_box

message_box('Colour Rendition Charts Computations')

message_box('Colour rendition charts chromaticity coordinates dataset.')
pprint(sorted(colour.COLOURCHECKERS.keys()))

print('\n')

message_box('Colour rendition charts spectral power distributions dataset.')
pprint(colour.COLOURCHECKERS_SPDS.keys())

print('\n')

message_box(('"ColorChecker 2005" colour rendition chart chromaticity '
             'coordinates data:\n'
             '\n\t("Patch Number", "Patch Name", "x", "y", "Y")'))
name, data, illuminant = colour.COLOURCHECKERS['ColorChecker 2005']
for index, name, x, y, Y in data:
    print(index, name, x, y, Y)

print('\n')

message_box(('Converting "ColorChecker 2005" colour rendition chart "CIE xyY" '
             'colourspace values to "sRGB" colourspace "RGB" values:\n'
             '\n\t("Patch Name", ["R", "G", "B"])'))
for index, name, x, y, Y in data:
    RGB = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(np.array([x, y, Y])),
        illuminant,
        colour.ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D65'],
        colour.sRGB_COLOURSPACE.XYZ_to_RGB_matrix,
        'Bradford',
        colour.sRGB_COLOURSPACE.encoding_cctf)

    RGB = [int(round(x * 255)) if x >= 0 else 0 for x in np.ravel(RGB)]
    print('"{0}": {1}'.format(name, RGB))
