# -*- coding: utf-8 -*-
"""
Showcases *Luminance* computations.
"""

import colour

from colour.utilities import message_box

message_box('"Luminance" Computations')

V = 3.74629715
message_box(('Computing "luminance" using '
             '"Newhall, Nickerson, and Judd (1943)" method for given '
             '"Munsell" value:\n'
             '\n\t{0}'.format(V)))
print(colour.luminance(V, method='Newhall 1943'))
print(colour.colorimetry.luminance_Newhall1943(V))

print('\n')

L = 37.98562910
message_box(('Computing "luminance" using "CIE 1976" method for given '
             '"Lightness":\n'
             '\n\t{0}'.format(L)))
print(colour.luminance(L))
print(colour.colorimetry.luminance_CIE1976(L))

print('\n')

L = 24.90229027
message_box(('Computing "luminance" using "Fairchild and Wyble (2010)" method '
             'for given "Lightness":\n'
             '\n\t{0}'.format(L)))
print(colour.luminance(L, method='Fairchild 2010') * 100)
print(colour.colorimetry.luminance_Fairchild2010(L) * 100)

print('\n')

L = 26.45950982
message_box(('Computing "luminance" using "Fairchild and Chen (2011)" method '
             'for given "Lightness":\n'
             '\n\t{0}'.format(L)))
print(colour.luminance(L, method='Fairchild 2011') * 100)
print(colour.colorimetry.luminance_Fairchild2011(L) * 100)

print('\n')

message_box(('Computing "luminance" using "ASTM D1535-08e1" method for given '
             '"Munsell" value:\n'
             '\n\t{0}'.format(V)))
print(colour.luminance(V, method='ASTM D1535-08'))
print(colour.colorimetry.luminance_ASTMD153508(V))
