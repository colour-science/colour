#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Munsell Renotation System* computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Munsell Renotation System" Computations')

Y = 10.08
message_box(('Computing "Munsell" value using '
             '"Priest, Gibson and MacNicholas (1920)" method for given '
             '"luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.munsell_value_Priest1920(Y))
print(colour.munsell_value(Y, method='Priest 1920'))

print('\n')

message_box(('Computing "Munsell" value using '
             '"Munsell, Sloan and Godlove (1933)" method for given '
             '"luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.munsell_value_Munsell1933(Y))
print(colour.munsell_value(Y, method='Munsell 1933'))

print('\n')

message_box(('Computing "Munsell" value using "Moon and Spencer (1943)" '
             'method for given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.munsell_value_Moon1943(Y))
print(colour.munsell_value(Y, method='Moon 1943'))

print('\n')

message_box(('Computing "Munsell" value using "Saunderson and Milner (1944)" '
             'method for given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.munsell_value_Saunderson1944(Y))
print(colour.munsell_value(Y, method='Saunderson 1944'))

print('\n')

message_box(('Computing "Munsell" value using "Ladd and Pinney (1955)" method '
             'for given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.munsell_value_Ladd1955(Y))
print(colour.munsell_value(Y, method='Ladd 1955'))

print('\n')

message_box(('Computing "Munsell" value using "McCamy (1987)" method for '
             'given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.munsell_value_McCamy1987(Y))
print(colour.munsell_value(Y, method='McCamy 1987'))

print('\n')

message_box(('Computing "Munsell" value using "ASTM D1535-08e1" method '
             'for given "luminance" value:\n'
             '\n\t{0}'.format(Y)))
print(colour.munsell_value_ASTMD153508(Y))
print(colour.munsell_value(Y, method='ASTM D1535-08'))

print('\n')

xyY = (0.38736945, 0.35751656, 0.59362000)
message_box(('Converting to "Munsell" colour from given "CIE xyY" '
             'colourspace values:\n'
             '\n\t{0}'.format(xyY)))
print(colour.xyY_to_munsell_colour(xyY))

print('\n')

for munsell_colour in ('4.2YR 8.1/5.3', 'N8.9'):
    message_box(('Converting to "CIE xyY" colourspace from given "Munsell" '
                 'colour:\n'
                 '\n\t{0}'.format(munsell_colour)))
    print(colour.munsell_colour_to_xyY(munsell_colour))
    print('\n')
