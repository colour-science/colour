# -*- coding: utf-8 -*-
"""
Showcases *Fairchild (1990)* chromatic adaptation model computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Fairchild (1990)" Chromatic Adaptation Model Computations')

XYZ_1 = np.array([0.1953, 0.2307, 0.2497])
XYZ_n = np.array([1.1115, 1.0000, 0.3520])
XYZ_r = np.array([0.9481, 1.0000, 1.0730])
Y_n = 200
message_box(('Computing chromatic adaptation using "Fairchild (1990)" '
             'chromatic adaptation model.\n'
             '\n\t"XYZ_1":\n\t\t{0}\n\t"XYZ_n":\n\t\t{1}\n\t"XYZ_r":\n\t\t{2}'
             '\n\t"Y_n":\n\t\t{3}\n\n'
             'Warning: The input domain and output range of that definition '
             'are non standard!'.format(XYZ_1, XYZ_n, XYZ_r, Y_n)))
print(
    colour.chromatic_adaptation(
        XYZ_1, XYZ_n, XYZ_r, method='Fairchild 1990', Y_n=Y_n))
print(
    colour.adaptation.chromatic_adaptation_Fairchild1990(
        XYZ_1 * 100.0, XYZ_n, XYZ_r, Y_n) / 100.0)
