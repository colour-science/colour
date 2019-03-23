# -*- coding: utf-8 -*-
"""
Showcases Look Up Table (LUT) data related examples.
"""

import numpy as np
import os

import colour
from colour.utilities import message_box

RESOURCES_DIRECTORY = os.path.join(
    os.path.dirname(__file__), '..', '..', 'io', 'luts', 'tests', 'resources')

message_box('Look Up Table (LUT) Data')

message_box('Reading "Iridas" ".cube" 3x1D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, 'iridas_cube',
                    'ACES_Proxy_10_to_ACES.cube')
print(colour.io.read_LUT_IridasCube(path))
print('\n')
print(colour.read_LUT(path))

print('\n')

message_box('Reading "Iridas" ".cube" 3D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, 'iridas_cube', 'ColourCorrect.cube')
print(colour.io.read_LUT_IridasCube(path))
print('\n')
print(colour.read_LUT(path))

print('\n')

message_box('Reading "Sony" ".spi1d" 1D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, 'sony_spi1d',
                    'oetf_reverse_sRGB_1D.spi1d')
print(colour.io.read_LUT_SonySPI1D(path))
print('\n')
print(colour.read_LUT(path))

print('\n')

message_box('Reading "Sony" ".spi1d" 3x1D LUT file.')
path = os.path.join(RESOURCES_DIRECTORY, 'sony_spi1d',
                    'oetf_reverse_sRGB_3x1D.spi1d')
print(colour.io.read_LUT_SonySPI1D(path))
print('\n')
print(colour.read_LUT(path))

print('\n')

RGB = np.array([0.35521588, 0.41000000, 0.24177934])
message_box(('Applying 1D LUT to given "RGB" values:\n' '\n\t{0}'.format(RGB)))
path = os.path.join(RESOURCES_DIRECTORY, 'sony_spi1d',
                    'oetf_reverse_sRGB_1D.spi1d')
LUT = colour.io.read_LUT(path)
print(LUT.apply(RGB))

print('\n')

message_box(('Applying 3x1D LUT to given "RGB" values:\n'
             '\n\t{0}'.format(RGB)))
path = os.path.join(RESOURCES_DIRECTORY, 'iridas_cube',
                    'ACES_Proxy_10_to_ACES.cube')
LUT = colour.io.read_LUT(path)
print(LUT.apply(RGB))

print('\n')

message_box(('Applying 3D LUT to given "RGB" values:\n' '\n\t{0}'.format(RGB)))
path = os.path.join(RESOURCES_DIRECTORY, 'iridas_cube', 'ColourCorrect.cube')
LUT = colour.io.read_LUT(path)
print(LUT.apply(RGB))
