# -*- coding: utf-8 -*-
"""
Showcases *LLAB(l:c)* colour appearance model computations.
"""

import numpy as np

import colour
from colour.appearance.llab import LLAB_ReferenceSpecification
from colour.utilities import message_box

message_box('"LLAB(l:c)" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_0 = np.array([95.05, 100.00, 108.88])
Y_b = 20.0
L = 318.31
surround = colour.LLAB_VIEWING_CONDITIONS['ref_average_4_minus']
message_box(
    ('Converting to "LLAB(l:c)" colour appearance model '
     'specification using given parameters:\n'
     '\n\tXYZ: {0}\n\tXYZ_0: {1}\n\tY_b: {2}\n\tL: {3}'
     '\n\tsurround: {4}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ, XYZ_0, Y_b, L, surround)))
specification = colour.XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)
print(specification)

print('\n')

message_box(('Broadcasting current output "LLAB(l:c)" colour appearance '
             'model specification to reference specification.\n'
             'The intent of this reference specification is to provide names '
             'as closest as possible to "Mark D. Fairchild" reference.\n'
             'The current output specification is meant to be consistent with '
             'the other colour appearance model specification by using same '
             'argument names for consistency wherever possible.'))

print(LLAB_ReferenceSpecification(*specification))
