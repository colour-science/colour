"""Showcases *Fairchild (1990)* chromatic adaptation model computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Fairchild (1990)" Chromatic Adaptation Model Computations')

XYZ_1 = np.array([0.1953, 0.2307, 0.2497])
XYZ_n = np.array([1.1115, 1.0000, 0.3520])
XYZ_r = np.array([0.9481, 1.0000, 1.0730])
Y_n = 200
message_box(
    f'Computing chromatic adaptation using "Fairchild (1990)" chromatic '
    f"adaptation model.\n\n"
    f'\t"XYZ_1": {XYZ_1}\n'
    f'\t"XYZ_n": {XYZ_n}\n'
    f'\t"XYZ_r": {XYZ_r}\n'
    f'\t"Y_n": {Y_n}'
)
print(
    colour.chromatic_adaptation(
        XYZ_1, XYZ_n, XYZ_r, method="Fairchild 1990", Y_n=Y_n
    )
)
print(
    colour.adaptation.chromatic_adaptation_Fairchild1990(
        XYZ_1 * 100, XYZ_n, XYZ_r, Y_n
    )
    / 100
)
