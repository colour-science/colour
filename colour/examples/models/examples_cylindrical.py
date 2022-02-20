"""Showcases cylindrical and spherical colour models computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box("Cylindrical & Spherical Colour Models")

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    f'Converting to the "HSV" colourspace from given "RGB" colourspace '
    f"values:\n\n\t{RGB}"
)
print(colour.RGB_to_HSV(RGB))

print("\n")

HSV = np.array([0.99603944, 0.93246304, 0.45620519])
message_box(
    f'Converting to the "RGB" colourspace from given "HSV" colourspace '
    f"values:\n\n\t{HSV}"
)
print(colour.HSV_to_RGB(HSV))

print("\n")

message_box(
    f'Converting to the "HSL" colourspace from given "RGB" colourspace '
    f"values:\n\n\t{RGB}"
)
print(colour.RGB_to_HSL(RGB))

print("\n")

HSL = np.array([0.99603944, 0.87347144, 0.24350795])
message_box(
    f'Converting to the "RGB" colourspace from given "HSL" colourspace '
    f"values:\n\n\t{HSL}"
)
print(colour.HSL_to_RGB(HSL))

print("\n")

message_box(
    f'Converting to the "HCL" colourspace from given "RGB" colourspace '
    f"values:\n\n\t{RGB}"
)
print(colour.RGB_to_HCL(RGB))

print("\n")

HCL = np.array([0.99603944, 0.87347144, 0.24350795])
message_box(
    f'Converting to the "RGB" colourspace from given "HCL" colourspace '
    f"values:\n\n\t{HCL}"
)
print(colour.HCL_to_RGB(HCL))

print("\n")
