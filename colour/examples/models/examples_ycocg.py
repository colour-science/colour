# -*- coding: utf-8 -*-
"""
Showcases *YCoCg* *colour encoding* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"YCoCg" Colour Encoding Computations')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(('Converting to "YCoCg" colour encoding from given "RGB" '
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_YCoCg(RGB))

print('\n')

YCoCg = np.array([0.13968653, 0.20764283, -0.10887582])
message_box(
    ('Converting to "RGB" colourspace values from "YCoCg" colour encoding '
     'values:\n'
     '\n\t{0}'.format(YCoCg)))
print(colour.YCoCg_to_RGB(YCoCg))
