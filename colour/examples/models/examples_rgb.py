"""Showcases *RGB* *colourspaces* computations."""

import numpy as np
from pprint import pprint

import colour
from colour.utilities import message_box

message_box('"RGB" Colourspaces Computations')

message_box('"RGB" colourspaces dataset.')
pprint(sorted(colour.RGB_COLOURSPACES.keys()))

print("\n")

message_box('"ACES2065-1" colourspaces data.')
colourspace = colour.RGB_COLOURSPACES["ACES2065-1"]
print(f'Name:\n"{colourspace.name}"')
print(f"\nPrimaries:\n{colourspace.primaries}")
print(
    f'\nNormalised primary matrix to "CIE XYZ" tristimulus values:\n'
    f"{colourspace.matrix_RGB_to_XYZ}"
)
print(
    f'\nNormalised primary matrix to "ACES2065-1":\n'
    f"{colourspace.matrix_XYZ_to_RGB}"
)
print(
    f"\nOpto-electronic transfer function from linear to colourspace:\n"
    f"{colourspace.cctf_encoding}"
)
print(
    f"\nElectro-optical transfer function from colourspace to linear:\n"
    f"{colourspace.cctf_decoding}"
)

print("\n")

message_box(
    'Computing the "ACES2065-1" colourspace to "ITU-R BT.709" colourspace '
    "matrix."
)
cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(
    colour.xy_to_XYZ(colourspace.whitepoint),
    colour.xy_to_XYZ(colour.RGB_COLOURSPACES["ITU-R BT.709"].whitepoint),
)
print(
    np.dot(
        colour.RGB_COLOURSPACES["ITU-R BT.709"].matrix_XYZ_to_RGB,
        np.dot(cat, colourspace.matrix_RGB_to_XYZ),
    )
)

print("\n")

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    f'Converting from the "ITU-R BT.709" colourspace to the "ACEScg" '
    f'colourspace given "RGB" values:\n\n\t{RGB}'
)
print(
    colour.RGB_to_RGB(
        RGB,
        colour.RGB_COLOURSPACES["ITU-R BT.709"],
        colour.RGB_COLOURSPACES["ACEScg"],
    )
)
