#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases overall *Colour* examples.
"""

import numpy as np
import warnings

import colour
from colour.utilities.verbose import message_box, warning

message_box('Filter "Colour" Warnings')

warning('This is a first warning and it can be filtered!')

colour.filter_warnings(True)

warning('This is a second warning and it has been filtered!')

colour.filter_warnings(False)

warning('This is a third warning and it has not been filtered!')

message_box('All Python can be filtered by setting the '
            '"colour.filter_warnings" definition "colour_warnings_only" '
            'argument.')

warnings.warn('This is a fourth warning and it has not been filtered!')

colour.filter_warnings(True, colour_warnings_only=False)

warning('This is a fifth warning and it has been filtered!')

colour.filter_warnings(False, colour_warnings_only=False)

warning('This is a sixth warning and it has not been filtered!')

print('\n')

message_box('Overall "Colour" Examples')

message_box('N-Dimensional Arrays Support')

XYZ = (0.07049534, 0.10080000, 0.09558313)
illuminant = (0.34570, 0.35850)
message_box('Using 1d "array_like" parameter:\n'
            '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

XYZ = np.tile(XYZ, (6, 1))
illuminant = np.tile(illuminant, (6, 1))
message_box('Using 2d "array_like" parameter:\n'
            '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

XYZ = np.reshape(XYZ, (2, 3, 3))
illuminant = np.reshape(illuminant, (2, 3, 2))
message_box('Using 3d "array_like" parameter:\n'
            '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

XYZ = np.reshape(XYZ, (3, 2, 1, 3))
illuminant = np.reshape(illuminant, (3, 2, 1, 2))
message_box('Using 4d "array_like" parameter:\n'
            '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

xy = np.tile((0.31270, 0.32900), (6, 1))
message_box(('Definitions return value may lose a dimension with respect to '
             'the parameter(s):\n'
             '\n{0}'.format(xy)))
print(colour.xy_to_CCT_McCamy1992(xy))

print('\n')

CCT = np.tile(6504.38938305, 6)
message_box(('Definitions return value may gain a dimension with respect to '
             'the parameter(s):\n'
             '\n{0}'.format(CCT)))
print(colour.CCT_to_xy_Kang2002(CCT))

print('\n')

message_box(('Definitions mixing "array_like" and "numeric" parameters '
             'expect the "numeric" parameters to have a dimension less than '
             'the "array_like" parameters.'))
XYZ_1 = (28.00, 21.26, 5.27)
xy_o1 = (0.4476, 0.4074)
xy_o2 = (0.3127, 0.3290)
Y_o = 20
E_o1 = 1000
E_o2 = 1000
message_box(('Parameters:\n\n'
             'XYZ_1:\n\n{0}\n\n'
             '\nxy_o1:\n\n{1}\n\n'
             '\nxy_o2:\n\n{2}\n\n'
             '\nY_o:\n\n{3}\n\n'
             '\nE_o1:\n\n{4}\n\n'
             '\nE_o2:\n\n{5}'.format(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(colour.chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

XYZ_1 = np.tile(XYZ_1, (6, 1))
message_box(('Parameters:\n\n'
             'XYZ_1:\n\n{0}\n\n'
             '\nxy_o1:\n\n{1}\n\n'
             '\nxy_o2:\n\n{2}\n\n'
             '\nY_o:\n\n{3}\n\n'
             '\nE_o1:\n\n{4}\n\n'
             '\nE_o2:\n\n{5}'.format(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(colour.chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

xy_o1 = np.tile(xy_o1, (6, 1))
xy_o2 = np.tile(xy_o2, (6, 1))
Y_o = np.tile(Y_o, 6)
E_o1 = np.tile(E_o1, 6)
E_o2 = np.tile(E_o2, 6)
message_box(('Parameters:\n\n'
             'XYZ_1:\n\n{0}\n\n'
             '\nxy_o1:\n\n{1}\n\n'
             '\nxy_o2:\n\n{2}\n\n'
             '\nY_o:\n\n{3}\n\n'
             '\nE_o1:\n\n{4}\n\n'
             '\nE_o2:\n\n{5}'.format(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(colour.chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
xy_o1 = np.reshape(xy_o1, (2, 3, 2))
xy_o2 = np.reshape(xy_o2, (2, 3, 2))
Y_o = np.reshape(Y_o, (2, 3))
E_o1 = np.reshape(E_o1, (2, 3))
E_o2 = np.reshape(E_o2, (2, 3))
message_box(('Parameters:\n\n'
             'XYZ_1:\n\n{0}\n\n'
             '\nxy_o1:\n\n{1}\n\n'
             '\nxy_o2:\n\n{2}\n\n'
             '\nY_o:\n\n{3}\n\n'
             '\nE_o1:\n\n{4}\n\n'
             '\nE_o2:\n\n{5}'.format(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(colour.chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))
