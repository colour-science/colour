#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Showcases *yellowness* computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Yellowness" Computations')

XYZ = (95.00000000, 100.00000000, 105.00000000)
message_box(('Computing "yellowness" using "ASTM D1925" method for '
             'given sample "CIE XYZ" tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.yellowness_ASTMD1925(XYZ))
print(colour.yellowness(XYZ=XYZ, method='ASTM D1925'))

print('\n')

message_box(('Computing "yellowness" using "ASTM E313" method for '
             'given sample "CIE XYZ" tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.yellowness_ASTME313(XYZ))
print(colour.yellowness(XYZ=XYZ, method='ASTM E313'))
