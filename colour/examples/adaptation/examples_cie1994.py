# -*- coding: utf-8 -*-
"""
Showcases *CIE 1994* chromatic adaptation model computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"CIE 1994" Chromatic Adaptation Model Computations')

XYZ_1 = np.array([0.2800, 0.2126, 0.0527])
xy_o1 = np.array([0.4476, 0.4074])
xy_o2 = np.array([0.3127, 0.3290])
Y_o = 20
E_o1 = 1000
E_o2 = 1000
message_box(('Computing chromatic adaptation using "CIE 1994" chromatic '
             'adaptation model.\n'
             '\n\t"XYZ_1":\n\t\t{0}\n\t"xy_o1":\n\t\t{1}\n\t"xy_o2":\n\t\t{2}'
             '\n\t"Y_o":\n\t\t{3}\n\t"E_o1":\n\t\t{4}'
             '\n\t"E_o2":\n\t\t{5}\n\n'
             'Warning: The input domain and output range of that definition '
             'are non standard!'.format(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)))
print(
    colour.chromatic_adaptation(
        XYZ_1,
        colour.xy_to_XYZ(xy_o1),
        colour.xy_to_XYZ(xy_o2),
        method='CIE 1994',
        Y_o=Y_o,
        E_o1=E_o1,
        E_o2=E_o2))
print(
    colour.adaptation.chromatic_adaptation_CIE1994(XYZ_1 * 100.0, xy_o1, xy_o2,
                                                   Y_o, E_o1, E_o2) / 100.0)
