# -*- coding: utf-8 -*-
"""
Showcases *Automatic Colour Conversion Graph* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('Automatic Colour Conversion Graph')

message_box('Converting a "ColorChecker" "dark skin" sample spectral '
            'distribution to "Output-Referred" "sRGB" colourspace.')

sd = colour.COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']
print(colour.convert(sd, 'Spectral Distribution', 'sRGB'))
print(
    colour.XYZ_to_sRGB(
        colour.sd_to_XYZ(sd, illuminant=colour.ILLUMINANTS_SDS['D65']) / 100))

print('\n')

RGB = np.array([0.45675795, 0.30986982, 0.24861924])
message_box(('Converting to "CAM16-UCS" colourspace from given '
             '"Output-Referred" "sRGB" colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.convert(RGB, 'Output-Referred RGB', 'CAM16UCS'))
specification = colour.XYZ_to_CAM16(
    colour.sRGB_to_XYZ(RGB) * 100,
    XYZ_w=colour.xy_to_XYZ(
        colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']) *
    100,
    L_A=64 / np.pi * 0.2,
    Y_b=20)
print(
    colour.JMh_CAM16_to_CAM16UCS(
        colour.utilities.tstack([
            specification.J,
            specification.M,
            specification.h,
        ])) / 100)

print('\n')

Jpapbp = np.array([0.39994811, 0.09206558, 0.0812752])
message_box(('Converting to "Output-Referred" "sRGB" colourspace from given '
             '"CAM16-UCS" colourspace colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(
    colour.convert(
        Jpapbp,
        'CAM16UCS',
        'sRGB',
        verbose_parameters={
            'describe': 'Extended',
            'width': 75
        }))
J, M, h = colour.utilities.tsplit(colour.CAM16UCS_to_JMh_CAM16(Jpapbp * 100))
specification = colour.CAM16_Specification(J=J, M=M, h=h)
print(
    colour.XYZ_to_sRGB(
        colour.CAM16_to_XYZ(
            specification,
            XYZ_w=colour.xy_to_XYZ(
                colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
                ['D65']) * 100,
            L_A=64 / np.pi * 0.2,
            Y_b=20) / 100))
