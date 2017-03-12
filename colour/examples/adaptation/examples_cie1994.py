#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *CIE 1994* chromatic adaptation model computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"CIE 1994" Chromatic Adaptation Model Computations')

XYZ_1 = (28.00, 21.26, 5.27)
xy_o1 = (0.4476, 0.4074)
xy_o2 = (0.3127, 0.3290)
Y_o = 20
E_o1 = 1000
E_o2 = 1000
message_box(('Computing chromatic adaptation using "CIE 1994" chromatic '
             'adaptation model.\n'
             '\n\t"XYZ_1":\n\t\t{0}\n\t"xy_o1":\n\t\t{1}\n\t"xy_o2":\n\t\t{2}'
             '\n\t"Y_o":\n\t\t{3}\n\t"E_o1":\n\t\t{4}'
             '\n\t"E_o2":\n\t\t{5}'.format(
                XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(colour.chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))
