"""Showcases *Nayatani (1995)* colour appearance model computations."""

import numpy as np

import colour
from colour.appearance.nayatani95 import CAM_ReferenceSpecification_Nayatani95
from colour.utilities import message_box

message_box('"Nayatani (1995)" Colour Appearance Model Computations')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_n = np.array([95.05, 100.00, 108.88])
Y_o = 20.0
E_o = 5000.0
E_or = 1000.0
message_box(
    f'Converting to the "Nayatani (1995)" colour appearance model '
    f"specification using given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_n: {XYZ_n}\n"
    f"\tY_o: {Y_o}\n"
    f"\tE_o: {E_o}\n"
    f"\tE_or: {E_or}"
)
specification = colour.XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)
print(specification)

print("\n")

message_box(
    'Broadcasting the current output "Nayatani (1995)" colour appearance '
    "model specification to the reference specification.\n"
    "The intent of this reference specification is to provide names "
    'as closest as possible to the "Mark D. Fairchild" reference.\n'
    "The current output specification is meant to be consistent with "
    "the other colour appearance model specification by using same "
    "argument names for consistency wherever possible."
)

print(CAM_ReferenceSpecification_Nayatani95(*specification.values))
