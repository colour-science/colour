"""Showcases *CMCCAT2000* chromatic adaptation model computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"CMCCAT200" Chromatic Adaptation Model Computations')

XYZ = np.array([0.2248, 0.2274, 0.0854])
XYZ_w = np.array([1.1115, 1.0000, 0.3520])
XYZ_wr = np.array([0.9481, 1.0000, 1.0730])
L_A1 = 200
L_A2 = 200
message_box(
    f'Computing chromatic adaptation using "CMCCAT200" forward chromatic '
    f"adaptation model.\n\n"
    f'\t"XYZ":\n\t\t{XYZ}\n'
    f'\t"XYZ_w":\n\t\t{XYZ_w}\n'
    f'\t"XYZ_wr":\n\t\t{XYZ_wr}\n'
    f'\t"L_A1":\n\t\t{L_A1}\n'
    f'\t"L_A2":\n\t\t{L_A2}'
)
print(
    colour.chromatic_adaptation(
        XYZ, XYZ_w, XYZ_wr, method="CMCCAT2000", L_A1=L_A1, L_A2=L_A2
    )
)
print(
    colour.adaptation.chromatic_adaptation_CMCCAT2000(
        XYZ * 100, XYZ_w, XYZ_wr, L_A1, L_A2
    )
    / 100
)

print("\n")

XYZ_c = np.array([0.19526983, 0.23068340, 0.24971752])
message_box(
    f'Computing chromatic adaptation using "CMCCAT200" inverse chromatic '
    f"adaptation model.\n\n"
    f'\t"XYZ_c": {XYZ_c}\n'
    f'\t"XYZ_w": {XYZ_w}\n'
    f'\t"XYZ_wr": {XYZ_wr}\n'
    f'\t"L_A1": {L_A1}\n'
    f'\t"L_A2": {L_A2}'
)
print(
    colour.chromatic_adaptation(
        XYZ_c,
        XYZ_w,
        XYZ_wr,
        method="CMCCAT2000",
        L_A1=L_A1,
        L_A2=L_A2,
        direction="Inverse",
    )
)
print(
    colour.adaptation.chromatic_adaptation_CMCCAT2000(
        XYZ_c * 100, XYZ_w, XYZ_wr, L_A1, L_A2, direction="Inverse"
    )
    / 100
)
