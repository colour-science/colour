"""
Showcases *Kim, Weyrich and Kautz (2009)* colour appearance model computations.
"""

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
    (
        'Converting to "Kim, Weyrich and Kautz (2009)" colour appearance model '
        "specification using given parameters:\n"
        "\n\tXYZ: {}\n\tXYZ_w: {}\n\tL_A: {}\n\tMedia: {}"
        "\n\tSurround: {}"
    ).format(XYZ, XYZ_w, L_A, media, surround)
)
specification = colour.XYZ_to_Kim2009(XYZ, XYZ_w, L_A, media, surround)
print(specification)

print("\n")

J = 28.861908975839647
C = 0.559245592437371
h = 219.048066776629530
specification = colour.CAM_Specification_Kim2009(J, C, h)
message_box(
    (
        'Converting to "CIE XYZ" tristimulus values using given '
        "parameters:\n"
        "\n\tJ: {}\n\tC: {}\n\th: {}\n\tXYZ_w: {}\n\tL_A: {}\n\tMedia: {}"
        "\n\tSurround: {}"
    ).format(J, C, h, XYZ_w, L_A, media, surround)
)
print(colour.Kim2009_to_XYZ(specification, XYZ_w, L_A, media, surround))
