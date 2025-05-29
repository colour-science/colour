"""
Demonstrate *Hunt* colour appearance model computations.

This module provides examples of colour appearance model computations using the
*Hunt* model, illustrating forward transformations from tristimulus values to
colour appearance correlates and reference specification broadcasting.
"""

import numpy as np

import colour
from colour.appearance.hunt import CAM_ReferenceSpecification_Hunt
from colour.utilities import message_box

message_box('Compute "Hunt" Colour Appearance Model Correlates')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
XYZ_b = np.array([95.05, 100.00, 108.88])
L_A = 318.31
surround = colour.VIEWING_CONDITIONS_HUNT["Normal Scenes"]
CCT_w = 6504.0
message_box(
    f'Convert to the "Hunt" colour appearance model specification using '
    f"given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tXYZ_b: {XYZ_b}\n"
    f"\tL_A: {L_A}\n"
    f"\tsurround: {surround}\n"
    f"\tCCT_w: {CCT_w}"
)

specification = colour.XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)
print(specification)

print("\n")

message_box(
    'Broadcast the current output "Hunt" colour appearance '
    "model specification to the reference specification.\n"
    "The intent of this reference specification is to provide names "
    'as closest as possible to the "Mark D. Fairchild" reference.\n'
    "The current output specification is meant to be consistent with "
    "the other colour appearance model specification by using same "
    "argument names for consistency wherever possible."
)

print(CAM_ReferenceSpecification_Hunt(*specification.values))
