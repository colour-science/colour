"""Showcases *Prismatic* colourspace computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Prismatic" Colourspace Computations')

RGB = np.array([0.25, 0.50, 0.75])
message_box(
    f'Converting from the "RGB" colourspace to the "Prismatic" colourspace '
    f'given "RGB" values:\n\n\t{RGB}'
)
print(colour.RGB_to_Prismatic(RGB))

print("\n")

Lrgb = np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
message_box(
    f'Converting from the "Prismatic" colourspace to the "RGB" colourspace '
    f'given "Lrgb" values:\n\n\t{Lrgb}'
)
print(colour.Prismatic_to_RGB(Lrgb))

print("\n")

message_box(
    f'Applying 50% desaturation in the "Prismatic" colourspace to the given '
    f'"RGB" values:\n\n\t{RGB}'
)
saturation = 0.5
Lrgb = colour.RGB_to_Prismatic(RGB)
Lrgb[..., 1:] = 1.0 / 3.0 + saturation * (Lrgb[..., 1:] - 1.0 / 3.0)
print(colour.Prismatic_to_RGB(Lrgb))
