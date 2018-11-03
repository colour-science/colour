# -*- coding: utf-8 -*-
"""
Showcases *ICTCP* *colour encoding* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"ICTCP" Colour Encoding Computations')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(('Converting from "ITU-R BT.2020" colourspace to "ICTCP" colour '
             'encoding given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_ICTCP(RGB))

print('\n')

ICTCP = np.array([0.07351364, 0.00475253, 0.09351596])
message_box(('Converting from "ICTCP" colour encoding to "ITU-R BT.2020" '
             'colourspace given "ICTCP" values:\n'
             '\n\t{0}'.format(ICTCP)))
print(colour.ICTCP_to_RGB(ICTCP))
