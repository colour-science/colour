#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *CMCCAT2000* chromatic adaptation model computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"CMCCAT200" Chromatic Adaptation Model Computations')

XYZ = (22.48, 22.74, 8.54)
XYZ_w = (111.15, 100.00, 35.20)
XYZ_wr = (94.81, 100.00, 107.30)
L_A1 = 200
L_A2 = 200
message_box(('Computing chromatic adaptation using "CMCCAT200" forward '
             'chromatic adaptation model.\n'
             '\n\t"XYZ":\n\t\t{0}\n\t"XYZ_w":\n\t\t{1}\n\t"XYZ_wr":\n\t\t{2}'
             '\n\t"L_A1":\n\t\t{3}\n\t"L_A2":\n\t\t{4}'.format(
                XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)))
print(colour.chromatic_adaptation_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

print('\n')

XYZ_c = (19.52698326, 23.06833960, 24.97175229)
message_box(('Computing chromatic adaptation using "CMCCAT200" reverse '
             'chromatic adaptation model.\n'
             '\n\t"XYZ_c":\n\t\t{0}\n\t"XYZ_w":\n\t\t{1}\n\t"XYZ_wr":\n\t\t{2}'
             '\n\t"L_A1":\n\t\t{3}\n\t"L_A2":\n\t\t{4}'.format(
                XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2)))
print(colour.chromatic_adaptation_CMCCAT2000(
    XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2, method='Reverse'))
