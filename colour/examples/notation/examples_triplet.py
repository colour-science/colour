#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases hexadecimal triplet computations.
"""

import colour.notation.triplet
from colour.utilities.verbose import message_box

message_box('Hexadecimal Triplet Computations')

RGB = (0.49019607843137253, 0.9803921568627451, 0.25098039215686274)
message_box(('Converting to "hex triplet" representation from given "RGB" '
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.notation.triplet.RGB_to_HEX(RGB))

print('\n')

hex_triplet = '#7dfa40'
message_box(('Converting to "RGB" colourspace from given "hex triplet" '
             'representation:\n'
             '\n\t{0}'.format(hex_triplet)))
print(colour.notation.triplet.HEX_to_RGB(hex_triplet))
