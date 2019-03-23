# -*- coding: utf-8 -*-
"""
Showcases colour rendition charts computations.
"""

import numpy as np
from pprint import pprint

import colour
from colour.utilities import message_box

message_box('Colour Rendition Charts Computations')

message_box('Colour rendition charts chromaticity coordinates dataset.')
pprint(sorted(colour.COLOURCHECKERS.keys()))

print('\n')

message_box('Colour rendition charts spectral distributions dataset.')
pprint(colour.COLOURCHECKERS_SDS.keys())

print('\n')

message_box(('"ColorChecker 2005" colour rendition chart chromaticity '
             'coordinates data:\n'
             '\n\t("Patch Number", "Patch Name", "xyY")'))
name, data, illuminant = colour.COLOURCHECKERS['ColorChecker 2005']
for name, xyY in data.items():
    print(name, xyY)

print('\n')

message_box(('Converting "ColorChecker 2005" colour rendition chart "CIE xyY" '
             'colourspace values to "sRGB" colourspace "RGB" values:\n'
             '\n\t("Patch Name", ["R", "G", "B"])'))
for name, xyY in data.items():
    RGB = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(xyY), illuminant,
        colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
        colour.RGB_COLOURSPACES['sRGB'].XYZ_to_RGB_matrix, 'Bradford',
        colour.RGB_COLOURSPACES['sRGB'].encoding_cctf)

    RGB = [int(round(x * 255)) if x >= 0 else 0 for x in np.ravel(RGB)]
    print('"{0}": {1}'.format(name, RGB))
