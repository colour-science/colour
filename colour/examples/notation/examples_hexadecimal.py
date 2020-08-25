# -*- coding: utf-8 -*-
"""
Showcases hexadecimal computations.
"""

import numpy as np

import colour.notation.hexadecimal
from colour.utilities import message_box

message_box('Hexadecimal Computations')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(('Converting to "hexadecimal" representation from given "RGB" '
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.notation.hexadecimal.RGB_to_HEX(RGB))

print('\n')

hex_triplet = '#74070a'
message_box(('Converting to "RGB" colourspace from given "hexadecimal" '
             'representation:\n'
             '\n\t{0}'.format(hex_triplet)))
print(colour.notation.hexadecimal.HEX_to_RGB(hex_triplet))
