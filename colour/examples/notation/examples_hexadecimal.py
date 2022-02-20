"""Showcases hexadecimal computations."""

import numpy as np

import colour.notation.hexadecimal
from colour.utilities import message_box

message_box("Hexadecimal Computations")

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    f'Converting to the "hexadecimal" representation from given "RGB"'
    f"colourspace values:\n\n\t{RGB}"
)
print(colour.notation.hexadecimal.RGB_to_HEX(RGB))

print("\n")

hex_triplet = "#74070a"
message_box(
    f'Converting to the "RGB" colourspace from given "hexadecimal" '
    f"representation:\n\n\t{hex_triplet}"
)
print(colour.notation.hexadecimal.HEX_to_RGB(hex_triplet))
