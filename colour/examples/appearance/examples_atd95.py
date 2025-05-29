"""
Demonstrate *ATD (1995)* colour appearance model computations.

This module provides examples of colour appearance model computations using the
*ATD (1995)* model, illustrating forward transformation from tristimulus values
to colour appearance correlates and reference specification broadcasting.
"""

import numpy as np

import colour
from colour.appearance.atd95 import CAM_ReferenceSpecification_ATD95
from colour.utilities import message_box

message_box('Compute "ATD (1995)" Colour Appearance Model Correlates')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_0 = np.array([95.05, 100.00, 108.88])
Y_0 = 318.31
k_1 = 0.0
k_2 = 50.0
message_box(
    f'Convert to the "ATD (1995)" colour appearance model specification '
    f"using given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_0: {XYZ_0}\n"
    f"\tY_0: {Y_0}\n"
    f"\tk_1: {k_1}\n"
    f"\tk_2: {k_2}"
)
specification = colour.XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)
print(specification)

print("\n")

message_box(
    'Broadcast the current output "ATD (1995)" colour appearance '
    "model specification to the reference specification.\n"
    "The intent of this reference specification is to provide names "
    'as closest as possible to the "Mark D. Fairchild" reference.\n'
    "The current output specification is meant to be consistent with "
    "the other colour appearance model specification by using same "
    "argument names for consistency wherever possible."
)

print(CAM_ReferenceSpecification_ATD95(*specification.values))
