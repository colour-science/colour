# -*- coding: utf-8 -*-
"""
Showcases *Lightness* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Lightness" Computations')

xyY = np.array([0.4316, 0.3777, 0.1008])
message_box(('Computing "Lightness" "CIE L*a*b*" reference value for given '
             '"CIE xyY" colourspace values:\n'
             '\n\t{0}'.format(xyY)))
print(np.ravel(colour.XYZ_to_Lab(colour.xyY_to_XYZ(xyY)))[0])

print('\n')

Y = 12.19722535
message_box(('Computing "Lightness" using '
             '"Glasser, Mckinney, Reilly and Schnelle (1958)" method for '
             'given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.lightness(Y, method='Glasser 1958'))
print(colour.colorimetry.lightness_Glasser1958(Y))

print('\n')

message_box(('Computing "Lightness" using "Wyszecki (1963)" method for '
             'given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.lightness(Y, method='Wyszecki 1963'))
print(colour.colorimetry.lightness_Wyszecki1963(Y))

print('\n')

message_box(('Computing "Lightness" using "Fairchild and Wyble (2010)" method '
             'for given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.lightness(Y, method='Fairchild 2010'))
print(colour.colorimetry.lightness_Fairchild2010(Y / 100.0))

print('\n')

message_box(('Computing "Lightness" using "Fairchild and Chen (2011)" method '
             'for given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.lightness(Y, method='Fairchild 2011'))
print(colour.colorimetry.lightness_Fairchild2011(Y / 100.0))

print('\n')

message_box(('Computing "Lightness" using "CIE 1976" method for '
             'given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.lightness(Y, method='CIE 1976'))
print(colour.colorimetry.lightness_CIE1976(Y))
