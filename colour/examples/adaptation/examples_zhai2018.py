# -*- coding: utf-8 -*-
"""
Showcases *Zhai and Luo (2018)* chromatic adaptation model computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Zhai and Luo (2018)" Chromatic Adaptation Model Computations')

XYZ_b = np.array([0.48900, 0.43620, 0.06250])
XYZ_wb = np.array([1.09850, 1, 0.35585])
XYZ_wd = np.array([0.95047, 1, 1.08883])
D_b = 0.9407
D_d = 0.9800
XYZ_wo = np.array([1, 1, 1])
message_box(
    (
        'Computing chromatic adaptation using "Zhai and Luo (2018)" '
        "chromatic adaptation model.\n"
        '\n\t"XYZ_b":\n\t\t{0}\n\t"XYZ_wb":\n\t\t{1}'
        '\n\t"XYZ_wd":\n\t\t{2}\n\t"D_b":\n\t\t{3}'
        '\n\t"D_d":\n\t\t{4}\n\t"XYZ_wo":\n\t\t{5}'.format(
            XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d, XYZ_wo
        )
    )
)
print(
    colour.chromatic_adaptation(
        XYZ_b,
        XYZ_wb,
        XYZ_wd,
        method="Zhai 2018",
        D_b=D_b,
        D_d=D_d,
        XYZ_wo=XYZ_wo,
    )
)
print(
    colour.adaptation.chromatic_adaptation_Zhai2018(
        XYZ_b * 100, XYZ_wb * 100, XYZ_wd * 100, D_b, D_d, XYZ_wo * 100
    )
    / 100
)
