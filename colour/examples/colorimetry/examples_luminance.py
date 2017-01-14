#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Luminance* computations.
"""

import colour

from colour.utilities.verbose import message_box

message_box('"Luminance" Computations')

V = 3.74629715
message_box(('Computing "luminance" using '
             '"Newhall, Nickerson, and Judd (1943)" method for given '
             '"Munsell" value:\n'
             '\n\t{0}'.format(V)))
print(colour.luminance_Newhall1943(V))
print(colour.luminance(V, method='Newhall 1943'))

print('\n')

L = 37.98562910
message_box(('Computing "luminance" using "CIE 1976" method for given '
             '"Lightness":\n'
             '\n\t{0}'.format(L)))
print(colour.luminance_CIE1976(L))
print(colour.luminance(L))

print('\n')

message_box(('Computing "luminance" using "ASTM D1535-08e1" method for given '
             '"Munsell" value:\n'
             '\n\t{0}'.format(V)))
print(colour.luminance_ASTMD153508(V))
print(colour.luminance(V, method='ASTM D1535-08'))
