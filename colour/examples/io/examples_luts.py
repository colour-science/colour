"""Showcases Look Up Table (LUT) data related examples."""

import numpy as np
import os

import colour
from colour.utilities import message_box

RESOURCES_DIRECTORY = os.path.join(
    os.path.dirname(__file__), "..", "..", "io", "luts", "tests", "resources"
)

message_box("Look Up Table (LUT) Data")

message_box('Reading a "Cinespace" ".csp" 3D LUT with Shaper file.')
path = os.path.join(
    RESOURCES_DIRECTORY, "cinespace", "Three_Dimensional_Table_With_Shaper.csp"
)
print(colour.io.read_LUT_Cinespace(path))
print("\n")
print(colour.read_LUT(path))

print("\n")

message_box('Reading an "Iridas" ".cube" 3x1D LUT file.')
path = os.path.join(
    RESOURCES_DIRECTORY, "iridas_cube", "ACES_Proxy_10_to_ACES.cube"
)
print(colour.io.read_LUT_IridasCube(path))
print("\n")
print(colour.read_LUT(path))

print("\n")

message_box('Reading an "Iridas" ".cube" 3D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, "iridas_cube", "Colour_Correct.cube")
print(colour.io.read_LUT_IridasCube(path))
print("\n")
print(colour.read_LUT(path))

print("\n")

message_box('Reading a "Sony" ".spi1d" 1D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, "sony_spi1d", "eotf_sRGB_1D.spi1d")
print(colour.io.read_LUT_SonySPI1D(path))
print("\n")
print(colour.read_LUT(path))

print("\n")

message_box('Reading a "Sony" ".spi1d" 3x1D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, "sony_spi1d", "eotf_sRGB_3x1D.spi1d")
print(colour.io.read_LUT_SonySPI1D(path))
print("\n")
print(colour.read_LUT(path))

print("\n")

message_box('Reading a "Sony" ".spi3d" 3D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, "sony_spi3d", "Colour_Correct.spi3d")
print(colour.io.read_LUT_SonySPI3D(path))
print("\n")
print(colour.read_LUT(path))

print("\n")

message_box('Reading a "Sony" ".spimtx" LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, "sony_spimtx", "dt.spimtx")
print(colour.io.read_LUT_SonySPImtx(path))
print("\n")
print(colour.read_LUT(path))

print("\n")

RGB = np.array([0.35521588, 0.41000000, 0.24177934])
message_box(f'Applying a 1D LUT to given "RGB" values:\n\n\t{RGB}')
path = os.path.join(RESOURCES_DIRECTORY, "sony_spi1d", "eotf_sRGB_1D.spi1d")
LUT = colour.io.read_LUT(path)
print(LUT.apply(RGB))

print("\n")

message_box(f'Applying a 3x1D LUT to given "RGB" values:\n\n\t{RGB}')
path = os.path.join(
    RESOURCES_DIRECTORY, "iridas_cube", "ACES_Proxy_10_to_ACES.cube"
)
LUT = colour.io.read_LUT(path)
print(LUT.apply(RGB))

print("\n")

message_box(f'Applying a 3D LUT to given "RGB" values:\n\n\t{RGB}')
path = os.path.join(RESOURCES_DIRECTORY, "iridas_cube", "Colour_Correct.cube")
LUT = colour.io.read_LUT(path)
print(LUT.apply(RGB))

message_box(
    f'Applying a "Sony" ".spimtx" LUT to given "RGB" values:\n\n\t{RGB}'
)
path = os.path.join(RESOURCES_DIRECTORY, "sony_spimtx", "dt.spimtx")
LUT = colour.io.read_LUT(path)
print(LUT.apply(RGB))
