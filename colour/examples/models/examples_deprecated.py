# -*- coding: utf-8 -*-
"""
Showcases deprecated colour models computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box(('Deprecated Colour Models Computations\n'
             '\nDon\'t use that! Seriously...'))

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(('Converting to "HSV" colourspace from given "RGB" colourspace '
             'values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_HSV(RGB))

print('\n')

HSV = np.array([0.99603944, 0.93246304, 0.45620519])
message_box(('Converting to "RGB" colourspace from given "HSV" colourspace '
             'values:\n'
             '\n\t{0}'.format(HSV)))
print(colour.HSV_to_RGB(HSV))

print('\n')

message_box(('Converting to "HSL" colourspace from given "RGB" colourspace '
             'values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_HSL(RGB))

print('\n')

HSL = np.array([0.99603944, 0.87347144, 0.24350795])
message_box(('Converting to "RGB" colourspace from given "HSL" colourspace '
             'values:\n'
             '\n\t{0}'.format(HSL)))
print(colour.HSL_to_RGB(HSL))

print('\n')

message_box(('Converting to "CMY" colourspace from given "RGB" colourspace '
             'values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_CMY(RGB))

print('\n')

CMY = np.array([0.54379481, 0.96918929, 0.95908048])
message_box(('Converting to "RGB" colourspace from given "CMY" colourspace '
             'values:\n'
             '\n\t{0}'.format(CMY)))
print(colour.CMY_to_RGB(CMY))

print('\n')

message_box(('Converting to "CMYK" colourspace from given "CMY" colourspace '
             'values:\n'
             '\n\t{0}'.format(CMY)))
print(colour.CMY_to_CMYK(CMY))

print('\n')

CMYK = np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])
message_box(('Converting to "CMY" colourspace from given "CMYK" colourspace '
             'values:\n'
             '\n\t{0}'.format(CMYK)))
print(colour.CMYK_to_CMY(CMYK))
