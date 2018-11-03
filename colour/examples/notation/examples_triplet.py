# -*- coding: utf-8 -*-
"""
Showcases hexadecimal triplet computations.
"""

import numpy as np

import colour.notation.triplet
from colour.utilities import message_box

message_box('Hexadecimal Triplet Computations')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(('Converting to "hex triplet" representation from given "RGB" '
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.notation.triplet.RGB_to_HEX(RGB))

print('\n')

hex_triplet = '#74070a'
message_box(('Converting to "RGB" colourspace from given "hex triplet" '
             'representation:\n'
             '\n\t{0}'.format(hex_triplet)))
print(colour.notation.triplet.HEX_to_RGB(hex_triplet))
