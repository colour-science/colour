"""Showcases *Von Kries* chromatic adaptation model computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Von Kries" Chromatic Adaptation Model Computations')

XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
message_box(
    f'Computing the chromatic adaptation matrix from two source "CIE XYZ" '
    f'tristimulus values arrays, default CAT is "CAT02".\n\n'
    f'\t"XYZ_w": {XYZ_w}\n'
    f'\t"XYZ_wr": {XYZ_wr}'
)
print(colour.adaptation.matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr))

print("\n")

message_box('Using "Bradford" CAT.')
print(
    colour.adaptation.matrix_chromatic_adaptation_VonKries(
        XYZ_w, XYZ_wr, transform="Bradford"
    )
)

print("\n")

message_box(
    "Computing the chromatic adaptation matrix from "
    'the "CIE Standard Illuminant A" to '
    'the "CIE Standard Illuminant D Series D65" using the "Von Kries" CAT.'
)
A = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["A"]
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
print(
    colour.adaptation.matrix_chromatic_adaptation_VonKries(
        colour.xy_to_XYZ(A), colour.xy_to_XYZ(D65), transform="Von Kries"
    )
)

print("\n")

XYZ = np.array([1.14176346, 1.00000000, 0.49815206])
message_box(
    f'Adapting given "CIE XYZ" tristimulus values from '
    f'the "CIE Standard Illuminant A" to the '
    f'"CIE Standard Illuminant D Series D65" using the "Sharp" CAT.\n\n'
    f'\t"XYZ": {XYZ}'
)
print(
    colour.chromatic_adaptation(
        XYZ, colour.xy_to_XYZ(A), colour.xy_to_XYZ(D65), transform="Sharp"
    )
)
print(
    colour.adaptation.chromatic_adaptation_VonKries(
        XYZ, colour.xy_to_XYZ(A), colour.xy_to_XYZ(D65), transform="Sharp"
    )
)
