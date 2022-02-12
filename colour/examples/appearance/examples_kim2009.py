"""Showcases *Kim, Weyrich and Kautz (2009)* colour appearance model computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box(
    '"Kim, Weyrich and Kautz (2009)" Colour Appearance Model Computations'
)

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31
media = colour.MEDIA_PARAMETERS_KIM2009["CRT Displays"]
surround = colour.VIEWING_CONDITIONS_KIM2009["Average"]
message_box(
    f'Converting to the "Kim, Weyrich and Kautz (2009)" colour appearance '
    f"model specification using given parameters:\n\n"
    f"\tXYZ: {XYZ}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tL_A: {L_A}\n"
    f"\tMedia: {media}\n"
    f"\tSurround: {surround}"
)
specification = colour.XYZ_to_Kim2009(XYZ, XYZ_w, L_A, media, surround)
print(specification)

print("\n")

J = 28.861908975839647
C = 0.559245592437371
h = 219.048066776629530
specification = colour.CAM_Specification_Kim2009(J, C, h)
message_box(
    f'Converting to "CIE XYZ" tristimulus values using given parameters:\n\n'
    f"\tJ: {J}\n"
    f"\tC: {C}\n"
    f"\th: {h}\n"
    f"\tXYZ_w: {XYZ_w}\n"
    f"\tL_A: {L_A}\n"
    f"\tMedia: {media}\n"
    f"\tSurround: {surround}"
)
print(colour.Kim2009_to_XYZ(specification, XYZ_w, L_A, media, surround))
