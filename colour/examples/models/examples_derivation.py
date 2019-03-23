# -*- coding: utf-8 -*-
"""
Showcases *RGB* colourspace derivation.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"RGB" Colourspace Derivation')

primaries = np.array([
    [0.73470, 0.26530],
    [0.00000, 1.00000],
    [0.00010, -0.07700],
])
whitepoint = np.array([0.32168, 0.33767])
message_box(('Computing the normalised primary matrix for "ACES2065-1" '
             'colourspace transforming from "ACES2065-1" colourspace to '
             '"CIE XYZ" tristimulus values using user defined primaries '
             'matrix and whitepoint:\n'
             '\n\t{0}\n\t{1}\n\t{2}\n\n\t{3}'.format(
                 primaries[0], primaries[1], primaries[2], whitepoint)))
print(colour.normalised_primary_matrix(primaries, whitepoint))

print('\n')

message_box(('Computing the normalised primary matrix for "ACES2065-1" '
             'colourspace transforming from "ACES2065-1" colourspace to '
             '"CIE XYZ" tristimulus values using colour models dataset.'))
print(
    colour.normalised_primary_matrix(
        colour.RGB_COLOURSPACES['ACES2065-1'].primaries,
        colour.RGB_COLOURSPACES['ACES2065-1'].whitepoint,
    ))

print('\n')

message_box(('Computing the normalised primary matrix for "ACES2065-1" '
             'colourspace transforming from "CIE XYZ" tristimulus values to '
             '"ACES2065-1" colourspace using colour models dataset.'))
print(
    np.linalg.inv(
        colour.normalised_primary_matrix(
            colour.RGB_COLOURSPACES['ACES2065-1'].primaries,
            colour.RGB_COLOURSPACES['ACES2065-1'].whitepoint,
        )))

print('\n')

message_box(('Computing "sRGB" colourspace primaries chromatically adapted to '
             '"CIE Standard Illuminant D50":\n'))
print(
    colour.chromatically_adapted_primaries(
        colour.RGB_COLOURSPACES['sRGB'].primaries,
        colour.RGB_COLOURSPACES['sRGB'].whitepoint,
        colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
    ))

print('\n')

npm = np.array([
    [0.41240000, 0.35760000, 0.18050000],
    [0.21260000, 0.71520000, 0.07220000],
    [0.01930000, 0.11920000, 0.95050000],
])
message_box(('Computing the primaries and whitepoint from given '
             'normalised primary matrix:\n'
             '\n{0}'.format(npm)))
print(colour.primaries_whitepoint(npm))

print('\n')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(('Computing "RGB" luminance of given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
print(
    colour.RGB_luminance(
        RGB,
        colour.RGB_COLOURSPACES['sRGB'].primaries,
        colour.RGB_COLOURSPACES['sRGB'].whitepoint,
    ))

print('\n')

message_box(('Computing "RGB" luminance equation of given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
print(
    colour.RGB_luminance(
        RGB,
        colour.RGB_COLOURSPACES['sRGB'].primaries,
        colour.RGB_COLOURSPACES['sRGB'].whitepoint,
    ))
