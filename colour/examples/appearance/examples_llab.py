"""Showcases *LLAB(l:c)* colour appearance model computations."""

import numpy as np

import colour
from colour.appearance.llab import CAM_ReferenceSpecification_LLAB
from colour.utilities import message_box

message_box('"LLAB(l:c)" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_0 = np.array([95.05, 100.00, 108.88])
Y_b = 20.0
L = 318.31
surround = colour.VIEWING_CONDITIONS_LLAB["ref_average_4_minus"]
message_box(
    f'Converting to the  "LLAB(l:c)" colour appearance model specification '
    f"using given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_0: {XYZ_0}\n"
    f"\tY_b: {Y_b}\n"
    f"\tL: {L}\n"
    f"\tsurround: {surround}"
)
specification = colour.XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)
print(specification)

print("\n")

message_box(
    'Broadcasting the current output "LLAB(l:c)" colour appearance '
    "model specification to the reference specification.\n"
    "The intent of this reference specification is to provide names "
    'as closest as possible to the "Mark D. Fairchild" reference.\n'
    "The current output specification is meant to be consistent with "
    "the other colour appearance model specification by using same "
    "argument names for consistency wherever possible."
)

print(CAM_ReferenceSpecification_LLAB(*specification.values))
