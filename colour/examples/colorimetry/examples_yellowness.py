# -*- coding: utf-8 -*-
"""
Showcases *yellowness* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Yellowness" Computations')

XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
message_box(
    ('Computing "yellowness" using "ASTM D1925" method for '
     'given sample "CIE XYZ" tristimulus values:\n'
     '\n\t{0}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ)))
print(colour.yellowness(XYZ=XYZ, method='ASTM D1925'))
print(colour.colorimetry.yellowness_ASTMD1925(XYZ))

print('\n')

message_box(
    ('Computing "yellowness" using "ASTM E313" method for '
     'given sample "CIE XYZ" tristimulus values:\n'
     '\n\t{0}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ)))
print(colour.yellowness(XYZ=XYZ, method='ASTM E313'))
print(colour.colorimetry.yellowness_ASTME313(XYZ))
