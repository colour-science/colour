# -*- coding: utf-8 -*-
"""
Showcases cylindrical and spherical colour models computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('Cylindrical & Spherical Colour Models')

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
