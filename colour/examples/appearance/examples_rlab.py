"""Showcases *RLAB* colour appearance model computations."""

import numpy as np

import colour
from colour.appearance.rlab import CAM_ReferenceSpecification_RLAB
from colour.utilities import message_box

message_box('"RLAB" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_n = np.array([109.85, 100, 35.58])
Y_n = 31.83
sigma = colour.VIEWING_CONDITIONS_RLAB["Average"]
D = colour.appearance.D_FACTOR_RLAB["Hard Copy Images"]
message_box(
    f'Converting to the "RLAB" colour appearance model specification using '
    f"given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_n: {XYZ_n}\n"
    f"\tY_n: {Y_n}\n"
    f"\tsigma: {sigma}\n"
    f"\tD: {D}"
)
specification = colour.XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)
print(specification)

print("\n")

message_box(
    'Broadcasting the current output "RLAB" colour appearance '
    "model specification to the reference specification.\n"
    "The intent of this reference specification is to provide names "
    'as closest as possible to the "Mark D. Fairchild" reference.\n'
    "The current output specification is meant to be consistent with "
    "the other colour appearance model specification by using same "
    "argument names for consistency wherever possible."
)

print(CAM_ReferenceSpecification_RLAB(*specification.values))
