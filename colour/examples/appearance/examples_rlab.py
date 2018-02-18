# -*- coding: utf-8 -*-
"""
Showcases *RLAB* colour appearance model computations.
"""

import numpy as np

import colour
from colour.appearance.rlab import RLAB_ReferenceSpecification
from colour.utilities import message_box

message_box('"RLAB" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_n = np.array([109.85, 100, 35.58])
Y_n = 31.83
sigma = colour.RLAB_VIEWING_CONDITIONS['Average']
D = colour.RLAB_D_FACTOR['Hard Copy Images']
message_box(
    ('Converting to "RLAB" colour appearance model '
     'specification using given parameters:\n'
     '\n\tXYZ: {0}\n\tXYZ_n: {1}\n\tY_n: {2}\n\tsigma: {3}'
     '\n\tD: {4}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ, XYZ_n, Y_n, sigma, D)))
specification = colour.XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)
print(specification)

print('\n')

message_box(('Broadcasting current output "RLAB" colour appearance '
             'model specification to reference specification.\n'
             'The intent of this reference specification is to provide names '
             'as closest as possible to "Mark D. Fairchild" reference.\n'
             'The current output specification is meant to be consistent with '
             'the other colour appearance model specification by using same '
             'argument names for consistency wherever possible.'))

print(RLAB_ReferenceSpecification(*specification))
