"""Showcases *YCoCg* *colour encoding* computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"YCoCg" Colour Encoding Computations')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    f'Converting to the "YCoCg" colour encoding from given "RGB" colourspace '
    f"values:\n\n\t{RGB}"
)
print(colour.RGB_to_YCoCg(RGB))

print("\n")

YCoCg = np.array([0.13968653, 0.20764283, -0.10887582])
message_box(
    f'Converting to the "RGB" colourspace values from "YCoCg" colour encoding '
    f"values:\n\n\t{YCoCg}"
)
print(colour.YCoCg_to_RGB(YCoCg))
