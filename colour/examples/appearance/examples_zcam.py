"""Showcases *ZCAM* colour appearance model computations."""

import numpy as np

import colour
from colour.appearance.zcam import CAM_ReferenceSpecification_ZCAM
from colour.utilities import message_box

message_box('"ZCAM" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_ZCAM["Average"]
message_box(
    f'Converting to the "ZCAM" colour appearance model specification using '
    f"given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tL_A: {L_A}\n"
    f"\tY_b: {Y_b}\n"
    f"\tSurround: {surround}"
)
specification = colour.XYZ_to_ZCAM(XYZ, XYZ_w, L_A, Y_b, surround)
print(specification)

print("\n")

message_box(
    'Broadcasting the current output "ZCAM" colour appearance model '
    "specification to the reference specification."
)

print(CAM_ReferenceSpecification_ZCAM(*specification.values))


print("\n")

J = 48.095883049811491
C = 0.18427174878137914
h = 219.74741565783773
specification = colour.CAM_Specification_ZCAM(J, C, h)
message_box(
    f'Converting to "CIE XYZ" tristimulus values using given parameters:\n\n'
    f"\tJ: {J}\n"
    f"\tC: {C}\n"
    f"\th: {h}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tL_A: {L_A}\n"
    f"\tY_b: {Y_b}\n"
    f"\tSurround: {surround}"
)
print(colour.ZCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))
