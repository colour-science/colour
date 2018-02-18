# -*- coding: utf-8 -*-
"""
Showcases *ATD (1995)* colour appearance model computations.
"""

import numpy as np

import colour
from colour.appearance.atd95 import ATD95_ReferenceSpecification
from colour.utilities import message_box

message_box('"ATD (1995)" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_0 = np.array([95.05, 100.00, 108.88])
Y_0 = 318.31
k_1 = 0.0
k_2 = 50.0
surround = colour.CIECAM02_VIEWING_CONDITIONS['Average']
message_box(
    ('Converting to "ATD (1995)" colour appearance model '
     'specification using given parameters:\n'
     '\n\tXYZ: {0}\n\tXYZ_0: {1}\n\tY_0: {2}\n\tk_1: {3}'
     '\n\tk_2: {4}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ, XYZ_0, Y_0, k_1, k_2)))
specification = colour.XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)
print(specification)

print('\n')

message_box(('Broadcasting current output "ATD (1995)" colour appearance '
             'model specification to reference specification.\n'
             'The intent of this reference specification is to provide names '
             'as closest as possible to "Mark D. Fairchild" reference.\n'
             'The current output specification is meant to be consistent with '
             'the other colour appearance model specification by using same '
             'argument names for consistency wherever possible.'))

print(ATD95_ReferenceSpecification(*specification))
