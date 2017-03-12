#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Delta E* colour difference computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Delta E" Computations')

Lab1 = (100.00000000, 21.57210357, 272.22819350)
Lab2 = (100.00000000, 426.67945353, 72.39590835)
message_box(('Computing "Delta E" with "CIE 1976" method from given "CIE Lab" '
             'colourspace matrices:\n'
             '\n\t{0}\n\t{1}'.format(Lab1, Lab2)))
print(colour.delta_E_CIE1976(Lab1, Lab2))
print(colour.delta_E(Lab1, Lab2, method='CIE 1976'))

print('\n')

message_box(('Computing "Delta E" with "CIE 1994" method from given "CIE Lab" '
             'colourspace matrices:\n'
             '\n\t{0}\n\t{1}'.format(Lab1, Lab2)))
print(colour.delta_E_CIE1994(Lab1, Lab2))
print(colour.delta_E(Lab1, Lab2, method='CIE 1994'))

print('\n')

message_box(('Computing "Delta E" with "CIE 1994" method from given "CIE Lab" '
             'colourspace matrices for "graphics arts" applications:\n'
             '\n\t{0}\n\t{1}'.format(Lab1, Lab2)))
print(colour.delta_E_CIE1994(Lab1, Lab2, textiles=False))
print(colour.delta_E(Lab1, Lab2, method='CIE 1994', textiles=False))

print('\n')

message_box(('Computing "Delta E" with "CIE 2000" method from given "CIE Lab" '
             'colourspace matrices:\n'
             '\n\t{0}\n\t{1}'.format(Lab1, Lab2)))
print(colour.delta_E_CIE2000(Lab1, Lab2))
print(colour.delta_E(Lab1, Lab2, method='CIE 2000'))

print('\n')

message_box(('Computing "Delta E" with "CMC" method from given "CIE Lab" '
             'colourspace matrices:\n'
             '\n\t{0}\n\t{1}'.format(Lab1, Lab2)))
print(colour.delta_E_CMC(Lab1, Lab2))
print(colour.delta_E(Lab1, Lab2, method='CMC'))

print('\n')

message_box(('Computing "Delta E" with "CMC" method from given "CIE Lab" '
             'colourspace matrices with imperceptibility threshold:\n'
             '\n\t{0}\n\t{1}'.format(Lab1, Lab2)))
print(colour.delta_E_CMC(Lab1, Lab2, l=1))
print(colour.delta_E(Lab1, Lab2, method='CMC', l=1))
