# -*- coding: utf-8 -*-
"""
Showcases overall *Colour* examples.
"""

import numpy as np
import warnings

import colour
from colour.utilities import (filter_warnings, message_box, warning,
                              runtime_warning, usage_warning)

message_box('Automatic Colour Conversion Graph')

message_box('Starting with version "0.3.14", "Colour" implements an automatic '
            'colour conversion graph enabling easier colour conversions.')

message_box('Converting a "ColorChecker" "dark skin" sample spectral '
            'distribution to "Output-Referred" "sRGB" colourspace.')

sd = colour.COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']

print(colour.convert(sd, 'Spectral Distribution', 'sRGB'))

print('\n')

RGB = np.array([0.45675795, 0.30986982, 0.24861924])
message_box(('Converting to "CAM16-UCS" colourspace from given '
             '"Output-Referred" "sRGB" colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.convert(RGB, 'Output-Referred RGB', 'CAM16UCS'))

print('\n')

Jpapbp = np.array([0.39994811, 0.09206558, 0.0812752])
message_box(('Converting to "Output-Referred" "sRGB" colourspace from given '
             '"CAM16-UCS" colourspace colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.convert(Jpapbp, 'CAM16UCS', 'sRGB'))

print('\n')

message_box('Filter "Colour" Warnings')

warning('This is a first warning and it can be filtered!')

filter_warnings()

warning('This is a second warning and it has been filtered!')

filter_warnings(False)

warning('This is a third warning and it has not been filtered!')

message_box('All Python can be filtered by setting the '
            '"colour.utilities.filter_warnings" definition "python_warnings" '
            'argument.')

warnings.warn('This is a fourth warning and it has not been filtered!')

filter_warnings(python_warnings=False)

warning('This is a fifth warning and it has been filtered!')

filter_warnings(False, python_warnings=False)

warning('This is a sixth warning and it has not been filtered!')

filter_warnings(False, python_warnings=False)

filter_warnings(colour_warnings=False, colour_runtime_warnings=True)

runtime_warning('This is a first runtime warning and it has been filtered!')

filter_warnings(colour_warnings=False, colour_usage_warnings=True)

usage_warning('This is a first usage warning and it has been filtered!')

print('\n')

message_box('Overall "Colour" Examples')

message_box('N-Dimensional Arrays Support')

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
illuminant = np.array([0.31270, 0.32900])
message_box('Using 1d "array_like" parameter:\n' '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

XYZ = np.tile(XYZ, (6, 1))
illuminant = np.tile(illuminant, (6, 1))
message_box('Using 2d "array_like" parameter:\n' '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

XYZ = np.reshape(XYZ, (2, 3, 3))
illuminant = np.reshape(illuminant, (2, 3, 2))
message_box('Using 3d "array_like" parameter:\n' '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

XYZ = np.reshape(XYZ, (3, 2, 1, 3))
illuminant = np.reshape(illuminant, (3, 2, 1, 2))
message_box('Using 4d "array_like" parameter:\n' '\n{0}'.format(XYZ))
print(colour.XYZ_to_Lab(XYZ, illuminant=illuminant))

print('\n')

xy = np.tile((0.31270, 0.32900), (6, 1))
message_box(('Definitions return value may lose a dimension with respect to '
             'the parameter(s):\n'
             '\n{0}'.format(xy)))
print(colour.xy_to_CCT(xy))

print('\n')

CCT = np.tile(6504.38938305, 6)
message_box(('Definitions return value may gain a dimension with respect to '
             'the parameter(s):\n'
             '\n{0}'.format(CCT)))
print(colour.CCT_to_xy(CCT))

print('\n')

message_box(('Definitions mixing "array_like" and "numeric" parameters '
             'expect the "numeric" parameters to have a dimension less than '
             'the "array_like" parameters.'))
XYZ_1 = np.array([28.00, 21.26, 5.27])
xy_o1 = np.array([0.4476, 0.4074])
xy_o2 = np.array([0.3127, 0.3290])
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
print(
    colour.adaptation.chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o,
                                                   E_o1, E_o2))

print('\n')

XYZ_1 = np.tile(XYZ_1, (6, 1))
message_box(('Parameters:\n\n'
             'XYZ_1:\n\n{0}\n\n'
             '\nxy_o1:\n\n{1}\n\n'
             '\nxy_o2:\n\n{2}\n\n'
             '\nY_o:\n\n{3}\n\n'
             '\nE_o1:\n\n{4}\n\n'
             '\nE_o2:\n\n{5}'.format(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(
    colour.adaptation.chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o,
                                                   E_o1, E_o2))

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
print(
    colour.adaptation.chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o,
                                                   E_o1, E_o2))

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
print(
    colour.adaptation.chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o,
                                                   E_o1, E_o2))

print('\n')

message_box('Domain-Range Scales')

message_box(('"Colour" uses two different domain-range scales: \n\n'
             '- "Reference"\n'
             '- "1"'))

print('\n')

message_box('Printing the current domain-range scale:')

print(colour.get_domain_range_scale())

print('\n')

message_box('Setting the current domain-range scale to "1":')

colour.set_domain_range_scale('1')

XYZ_1 = np.array([0.2800, 0.2126, 0.0527])
xy_o1 = np.array([0.4476, 0.4074])
xy_o2 = np.array([0.3127, 0.3290])
Y_o = 0.2
E_o1 = 1000
E_o2 = 1000
message_box(('Parameters:\n\n'
             'XYZ_1:\n\n{0}\n\n'
             '\nxy_o1:\n\n{1}\n\n'
             '\nxy_o2:\n\n{2}\n\n'
             '\nY_o:\n\n{3}\n\n'
             '\nE_o1:\n\n{4}\n\n'
             '\nE_o2:\n\n{5}'.format(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(
    colour.adaptation.chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o,
                                                   E_o1, E_o2))
