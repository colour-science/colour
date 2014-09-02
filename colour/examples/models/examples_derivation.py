#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *RGB* colourspace derivation.
"""

import numpy as np

import colour
from colour.utilities.verbose import message_box

message_box('"RGB" Colourspace Derivation')

primaries = np.array(
    [[0.73470, 0.26530],
     [0.00000, 1.00000],
     [0.00010, -0.07700]])
whitepoint = (0.32168, 0.33767)
message_box(('Computing the normalised primary matrix for "ACES RGB" '
             'colourspace transforming from "ACES RGB" colourspace to '
             '"CIE XYZ" colourspace using user defined primaries matrix and '
             'whitepoint:\n'
             '\n\t{0}\n\t{1}\n\t{2}\n\n\t{3}'.format(primaries[0],
                                                     primaries[1],
                                                     primaries[2],
                                                     whitepoint)))
print(colour.normalised_primary_matrix(primaries, whitepoint))

print('\n')

message_box(('Computing the normalised primary matrix for "ACES RGB" '
             'colourspace transforming from "ACES RGB" colourspace to '
             '"CIE XYZ" colourspace using colour models dataset.'))
print(colour.normalised_primary_matrix(
    colour.ACES_RGB_COLOURSPACE.primaries,
    colour.ACES_RGB_COLOURSPACE.whitepoint))

print('\n')

message_box(('Computing the normalised primary matrix for "ACES RGB" '
             'colourspace transforming from "CIE XYZ" colourspace to '
             '"ACES RGB" colourspace using colour models dataset.'))
print(np.linalg.inv(colour.normalised_primary_matrix(
    colour.ACES_RGB_COLOURSPACE.primaries,
    colour.ACES_RGB_COLOURSPACE.whitepoint)))

print('\n')

RGB = [56, 16, 100]
message_box(('Computing the normalised primary matrix for "RGB" luminance of '
             'given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_luminance(
    RGB,
    colour.sRGB_COLOURSPACE.primaries,
    colour.sRGB_COLOURSPACE.whitepoint))
