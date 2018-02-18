# -*- coding: utf-8 -*-
"""
Showcases *Nayatani (1995)* colour appearance model computations.
"""

import numpy as np

import colour
from colour.appearance.nayatani95 import Nayatani95_ReferenceSpecification
from colour.utilities import message_box

message_box('"Nayatani (1995)" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_n = np.array([95.05, 100.00, 108.88])
Y_o = 20.0
E_o = 5000.0
E_or = 1000.0
message_box(
    ('Converting to "Nayatani (1995)" colour appearance model '
     'specification using given parameters:\n'
     '\n\tXYZ: {0}\n\tXYZ_n: {1}\n\tY_o: {2}\n\tE_o: {3}'
     '\n\tE_or: {4}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ, XYZ_n, Y_o, E_o, E_or)))
specification = colour.XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)
print(specification)

print('\n')

message_box(('Broadcasting current output "Nayatani (1995)" colour appearance '
             'model specification to reference specification.\n'
             'The intent of this reference specification is to provide names '
             'as closest as possible to "Mark D. Fairchild" reference.\n'
             'The current output specification is meant to be consistent with '
             'the other colour appearance model specification by using same '
             'argument names for consistency wherever possible.'))

print(Nayatani95_ReferenceSpecification(*specification))
