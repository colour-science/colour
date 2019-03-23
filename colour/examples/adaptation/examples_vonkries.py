# -*- coding: utf-8 -*-
"""
Showcases *Von Kries* chromatic adaptation model computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Von Kries" Chromatic Adaptation Model Computations')

XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
message_box(('Computing the chromatic adaptation matrix from two source '
             '"CIE XYZ" tristimulus values arrays, default CAT is "CAT02".\n'
             '\n\t"XYZ_w":\n\t\t{0}\n\t"XYZ_wr":\n\t\t{1}'.format(
                 XYZ_w, XYZ_wr)))
print(colour.adaptation.chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr))

print('\n')

message_box('Using "Bradford" CAT.')
print(
    colour.adaptation.chromatic_adaptation_matrix_VonKries(
        XYZ_w, XYZ_wr, transform='Bradford'))

print('\n')

message_box(('Computing the chromatic adaptation matrix from '
             '"CIE Standard Illuminant A" to '
             '"CIE Standard Illuminant D Series D65" using "Von Kries" CAT.'))
A = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['A']
D65 = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
print(
    colour.adaptation.chromatic_adaptation_matrix_VonKries(
        colour.xy_to_XYZ(A), colour.xy_to_XYZ(D65), transform='Von Kries'))

print('\n')

XYZ = np.array([1.14176346, 1.00000000, 0.49815206])
message_box(('Adapting given "CIE XYZ" tristimulus values from '
             '"CIE Standard Illuminant A" to '
             '"CIE Standard Illuminant D Series D65" using "Sharp" CAT.\n'
             '\n\t"XYZ":\n\t\t{0}'.format(XYZ)))
print(
    colour.chromatic_adaptation(
        XYZ, colour.xy_to_XYZ(A), colour.xy_to_XYZ(D65), transform='Sharp'))
print(
    colour.adaptation.chromatic_adaptation_VonKries(
        XYZ, colour.xy_to_XYZ(A), colour.xy_to_XYZ(D65), transform='Sharp'))
