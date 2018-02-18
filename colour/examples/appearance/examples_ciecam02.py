# -*- coding: utf-8 -*-
"""
Showcases *CIECAM02* colour appearance model computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"CIECAM02" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31
Y_b = 20.0
surround = colour.CIECAM02_VIEWING_CONDITIONS['Average']
message_box(
    ('Converting to "CIECAM02" colour appearance model specification '
     'using given parameters:\n'
     '\n\tXYZ: {0}\n\tXYZ_w: {1}\n\tL_A: {2}\n\tY_b: {3}'
     '\n\tSurround: {4}\n\n'
     'Warning: The input domain of that definition is non standard!').format(
         XYZ, XYZ_w, L_A, Y_b, surround))
specification = colour.XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
print(specification)

print('\n')

J = 41.73109113
C = 0.10470776
h = 219.04843266
specification = colour.CIECAM02_Specification(J, C, h)
message_box(
    ('Converting to "CIE XYZ" tristimulus values using given '
     'parameters:\n'
     '\n\tJ: {0}\n\tC: {1}\n\th: {2}\n\tXYZ_w: {3}\n\tL_A: {4}'
     '\n\tY_b: {5}\n\n'
     'Warning: The output range of that definition is non standard!').format(
         J, C, h, XYZ_w, L_A, Y_b))
print(colour.CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b))
