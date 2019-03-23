# -*- coding: utf-8 -*-
"""
Showcases *RGB* *colourspaces* computations.
"""

import numpy as np
from pprint import pprint

import colour
from colour.utilities import message_box

message_box('"RGB" Colourspaces Computations')

message_box('"RGB" colourspaces dataset.')
pprint(sorted(colour.RGB_COLOURSPACES.keys()))

print('\n')

message_box('"ACES2065-1" colourspaces data.')
colourspace = colour.RGB_COLOURSPACES['ACES2065-1']
print('Name:\n"{0}"'.format(colourspace.name))
print('\nPrimaries:\n{0}'.format(colourspace.primaries))
print(('\nNormalised primary matrix to "CIE XYZ" '
       'tristimulus values:\n{0}').format(colourspace.RGB_to_XYZ_matrix))
print('\nNormalised primary matrix to "ACES2065-1":\n{0}'.format(
    colourspace.XYZ_to_RGB_matrix))
print('\nOpto-electronic transfer function from '
      'linear to colourspace:\n{0}'.format(colourspace.encoding_cctf))
print('\nElectro-optical transfer function from '
      'colourspace to linear:\n{0}'.format(colourspace.decoding_cctf))

print('\n')

message_box(
    ('Computing "ACES2065-1" colourspace to "ITU-R BT.709" colourspace '
     'matrix.'))
cat = colour.chromatic_adaptation_matrix_VonKries(
    colour.xy_to_XYZ(colourspace.whitepoint),
    colour.xy_to_XYZ(colour.RGB_COLOURSPACES['ITU-R BT.709'].whitepoint))
print(
    np.dot(colour.RGB_COLOURSPACES['ITU-R BT.709'].XYZ_to_RGB_matrix,
           np.dot(cat, colourspace.RGB_to_XYZ_matrix)))

print('\n')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    ('Converting from "ITU-R BT.709" colourspace to "ACEScg" colourspace '
     'given "RGB" values:\n'
     '\n\t{0}'.format(RGB)))
print(
    colour.RGB_to_RGB(
        RGB,
        colour.RGB_COLOURSPACES['ITU-R BT.709'],
        colour.RGB_COLOURSPACES['ACEScg'],
    ))
