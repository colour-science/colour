"""Showcases Cyan-Magenta-Yellow (Black) (CMY(K)) colour transformations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box("Cyan-Magenta-Yellow (Black) (CMY(K)) Colour Transformations")

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    f'Converting to the "CMY" colourspace from given "RGB" colourspace '
    f"values:\n\n\t{RGB}"
)
print(colour.RGB_to_CMY(RGB))

print("\n")

CMY = np.array([0.54379481, 0.96918929, 0.95908048])
message_box(
    f'Converting to the "RGB" colourspace from given "CMY" colourspace '
    f"values:\n\n\t{CMY}"
)
print(colour.CMY_to_RGB(CMY))

print("\n")

message_box(
    f'Converting to the "CMYK" colourspace from given "CMY" colourspace '
    f"values:\n\n\t{CMY}"
)
print(colour.CMY_to_CMYK(CMY))

print("\n")

CMYK = np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])
message_box(
    f'Converting to the "CMY" colourspace from given "CMYK" colourspace '
    f"values:\n\n\t{CMYK}"
)
print(colour.CMYK_to_CMY(CMYK))
