#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *ICTCP* *colour encoding* computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"ICTCP" Colour Encoding Computations')

RGB = (0.35181454, 0.26934757, 0.21288023)
message_box(('Converting from "Rec. 2020" colourspace to "ICTCP" colour '
             'encoding given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_ICTCP(RGB))

print('\n')

ICTCP = (0.09554079, -0.00890639, 0.01389286)
message_box(('Converting from "ICTCP" colour encoding to "Rec. 2020" '
             'colourspace given "ICTCP" values:\n'
             '\n\t{0}'.format(ICTCP)))
print(colour.ICTCP_to_RGB(ICTCP))
