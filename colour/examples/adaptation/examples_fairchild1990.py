#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Fairchild (1990)* chromatic adaptation model computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Fairchild (1990)" Chromatic Adaptation Model Computations')

XYZ_1 = (19.53, 23.07, 24.97)
XYZ_n = (111.15, 100.00, 35.20)
XYZ_r = (94.81, 100.00, 107.30)
Y_n = 200
message_box(('Computing chromatic adaptation using "Fairchild (1990)" '
             'chromatic adaptation model.\n'
             '\n\t"XYZ_1":\n\t\t{0}\n\t"XYZ_n":\n\t\t{1}\n\t"XYZ_r":\n\t\t{2}'
             '\n\t"Y_n":\n\t\t{3}'.format(XYZ_1, XYZ_n, XYZ_r, Y_n)))
print(colour.chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n))
