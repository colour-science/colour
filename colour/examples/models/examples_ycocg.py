# -*- coding: utf-8 -*-
"""
Showcases *YCoCg* *colour encoding* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"YCoCg" Colour Encoding Computations')

RGB = np.array([0.75, 0.75, 0.0])
message_box(('Converting to "YCoCg" colour encoding from given "RGB" ' 
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_YCoCg(RGB))

print('\n')

YCoCg = np.array([0.5625, 0.375, 0.1875])
message_box(
    ('Converting to "RGB" colourspace values from "YCoCg" colour encoding '
     'values:\n'
     '\n\t{0}'.format(YCoCg)))
print(colour.YCoCg_to_RGB(YCoCg))
