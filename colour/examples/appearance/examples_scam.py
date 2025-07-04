"""
Demonstrate *sCAM* colour appearance model computations.

This module provides examples of colour appearance model computations using the
*sCAM* (Simple Colour Appearance Model), illustrating both forward and inverse
transformations between tristimulus values and colour appearance correlates.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('Compute "sCAM" Colour Appearance Model Correlates')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_sCAM["Average"]
message_box(
    f'Convert to the "sCAM" colour appearance model specification '
    f"using given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tL_A: {L_A}\n"
    f"\tY_b: {Y_b}\n"
    f"\tSurround: {surround}"
)
specification = colour.XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround)
print(specification)

print("\n")

J = 49.9795668
C = 0.0140531
h = 328.2724924
specification = colour.CAM_Specification_sCAM(J, C, h)
message_box(
    f'Convert to "CIE XYZ" tristimulus values using given parameters:\n\n'
    f"\tJ: {J}\n"
    f"\tC: {C}\n"
    f"\th: {h}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tL_A: {L_A}\n"
    f"\tY_b: {Y_b}\n"
    f"\tSurround: {surround}"
)
print(colour.sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

print("\n")

message_box('Using "colourfulness" instead of "chroma"')
M = 0.0050244
specification = colour.CAM_Specification_sCAM(J, None, h, M=M)
message_box(
    f'Convert to "CIE XYZ" tristimulus values using given parameters:\n\n'
    f"\tJ: {J}\n"
    f"\tM: {M}\n"
    f"\th: {h}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tL_A: {L_A}\n"
    f"\tY_b: {Y_b}\n"
    f"\tSurround: {surround}"
)
print(colour.sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

print("\n")

message_box("Using discount illuminant")
specification = colour.XYZ_to_sCAM(
    XYZ, XYZ_w, L_A, Y_b, surround, discount_illuminant=True
)
print(specification)
