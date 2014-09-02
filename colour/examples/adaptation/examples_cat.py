#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases chromatic adaptation computations.
"""

from __future__ import division, unicode_literals

import colour
from colour.utilities.verbose import message_box

message_box('Chromatic Adaptation Computations')

XYZ1 = (1.09923822, 1.000, 0.35445412)
XYZ2 = (0.96907232, 1.000, 1.121792157)
message_box(('Computing the chromatic adaptation matrix from two source '
             '"CIE XYZ" matrices, default CAT is "CAT02".\n'
             '\n\t"XYZ1":\n\t\t{0}\n\t"XYZ2":\n\t\t{1}'.format(XYZ1, XYZ2)))
print(colour.chromatic_adaptation_matrix(XYZ1, XYZ2))

print('\n')

message_box('Using "Bradford" CAT.')
print(colour.chromatic_adaptation_matrix(XYZ1, XYZ2, method='Bradford'))

print('\n')

message_box(('Computing the chromatic adaptation matrix from '
             '"CIE Standard Illuminant A" to '
             '"CIE Standard Illuminant D Series D60" using "Von Kries" CAT.'))
A = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['A']
D60 = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D60']

print(colour.chromatic_adaptation_matrix(
    colour.xy_to_XYZ(A),
    colour.xy_to_XYZ(D60),
    method='Von Kries'))
