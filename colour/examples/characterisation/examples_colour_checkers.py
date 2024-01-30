"""Showcases colour rendition charts computations."""

from pprint import pprint

import numpy as np

import colour
from colour.utilities import message_box

message_box("Colour Rendition Charts Computations")

message_box("Colour rendition charts chromaticity coordinates dataset.")
pprint(sorted(colour.CCS_COLOURCHECKERS.keys()))

print("\n")

message_box("Colour rendition charts spectral distributions dataset.")
pprint(colour.SDS_COLOURCHECKERS.keys())

print("\n")

message_box(
    '"ColorChecker 2005" colour rendition chart chromaticity coordinates data:\n\n'
    '\t("Patch Number", "Patch Name", "xyY")'
)
name, data, illuminant, rows, columns = colour.CCS_COLOURCHECKERS["ColorChecker 2005"]
for name, xyY in data.items():
    print(name, xyY)

print("\n")

message_box(
    'Converting the "ColorChecker 2005" colour rendition chart "CIE xyY" '
    'colourspace values to "sRGB" colourspace "RGB" values:\n\n'
    '\t("Patch Name", ["R", "G", "B"])'
)
for name, xyY in data.items():
    RGB = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(xyY),
        colour.RGB_COLOURSPACES["sRGB"],
        illuminant,
        "Bradford",
        apply_cctf_encoding=True,
    )

    RGB_i = [int(round(x * 255)) if x >= 0 else 0 for x in np.ravel(RGB)]
    print(f'"{name}": {RGB_i}')
