# -*- coding: utf-8 -*-
"""
Showcases *Hunt* colour appearance model computations.
"""

import numpy as np

import colour
from colour.appearance.hunt import Hunt_ReferenceSpecification
from colour.utilities import message_box

message_box('"Hunt" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
XYZ_b = np.array([95.05, 100.00, 108.88])
L_A = 318.31
surround = colour.HUNT_VIEWING_CONDITIONS['Normal Scenes']
CCT_w = 6504.0
message_box(
    ('Converting to "Hunt" colour appearance model '
     'specification using given parameters:\n'
     '\n\tXYZ: {0}\n\tXYZ_w: {1}\n\tXYZ_b: {2}\n\tL_A: {3}'
     '\n\tsurround: {4}\n\tCCT_w: {5}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w)))

specification = colour.XYZ_to_Hunt(
    XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)
print(specification)

print('\n')

message_box(('Broadcasting current output "Hunt" colour appearance '
             'model specification to reference specification.\n'
             'The intent of this reference specification is to provide names '
             'as closest as possible to "Mark D. Fairchild" reference.\n'
             'The current output specification is meant to be consistent with '
             'the other colour appearance model specification by using same '
             'argument names for consistency wherever possible.'))

print(Hunt_ReferenceSpecification(*specification))
