# -*- coding: utf-8 -*-
"""
Showcases *CMCCAT2000* chromatic adaptation model computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"CMCCAT200" Chromatic Adaptation Model Computations')

XYZ = np.array([0.2248, 0.2274, 0.0854])
XYZ_w = np.array([1.1115, 1.0000, 0.3520])
XYZ_wr = np.array([0.9481, 1.0000, 1.0730])
L_A1 = 200
L_A2 = 200
message_box(('Computing chromatic adaptation using "CMCCAT200" forward '
             'chromatic adaptation model.\n'
             '\n\t"XYZ":\n\t\t{0}\n\t"XYZ_w":\n\t\t{1}\n\t"XYZ_wr":\n\t\t{2}'
             '\n\t"L_A1":\n\t\t{3}\n\t"L_A2":\n\t\t{4}\n\n'
             'Warning: The input domain and output range of that definition '
             'are non standard!'.format(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)))
print(
    colour.chromatic_adaptation(
        XYZ, XYZ_w, XYZ_wr, method='CMCCAT2000', L_A1=L_A1, L_A2=L_A2))
print(
    colour.adaptation.chromatic_adaptation_CMCCAT2000(
        XYZ * 100.0, XYZ_w, XYZ_wr, L_A1, L_A2) / 100.0)

print('\n')

XYZ_c = np.array([0.19526983, 0.23068340, 0.24971752])
message_box(('Computing chromatic adaptation using "CMCCAT200" reverse '
             'chromatic adaptation model.\n'
             '\n\t"XYZ_c":\n\t\t{0}\n\t"XYZ_w":\n\t\t{1}\n\t"XYZ_wr":\n\t\t{2}'
             '\n\t"L_A1":\n\t\t{3}\n\t"L_A2":\n\t\t{4}\n\n'
             'Warning: The input domain and output range of that definition '
             'are non standard!'.format(XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2)))
print(
    colour.chromatic_adaptation(
        XYZ_c,
        XYZ_w,
        XYZ_wr,
        method='CMCCAT2000',
        L_A1=L_A1,
        L_A2=L_A2,
        direction='Reverse'))
print(
    colour.adaptation.chromatic_adaptation_CMCCAT2000(
        XYZ_c * 100.0, XYZ_w, XYZ_wr, L_A1, L_A2, direction='Reverse') / 100.0)
