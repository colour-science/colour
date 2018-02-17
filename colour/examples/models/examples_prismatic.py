# -*- coding: utf-8 -*-
"""
Showcases *Prismatic* colourspace computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Prismatic" Colourspace Computations')

RGB = np.array([0.25, 0.50, 0.75])
message_box(('Converting from "RGB" colourspace to "Prismatic" colourspace '
             'given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_Prismatic(RGB))

print('\n')

Lrgb = np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
message_box(('Converting from "Prismatic" colourspace to "RGB" colourspace '
             'given "Lrgb" values:\n'
             '\n\t{0}'.format(Lrgb)))
print(colour.Prismatic_to_RGB(Lrgb))

print('\n')

message_box(('Applying 50% desaturation in "Prismatic" colourspace to'
             'given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
saturation = 0.5
Lrgb = colour.RGB_to_Prismatic(RGB)
Lrgb[..., 1:] = 1.0 / 3.0 + saturation * (Lrgb[..., 1:] - 1.0 / 3.0)
print(colour.Prismatic_to_RGB(Lrgb))
