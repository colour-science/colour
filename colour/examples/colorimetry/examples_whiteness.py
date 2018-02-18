# -*- coding: utf-8 -*-
"""
Showcases *whiteness* computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Whiteness" Computations')

XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
message_box(
    ('Computing "whiteness" using "Berger (1959)" method for '
     'given sample and reference white "CIE XYZ" tristimulus values '
     'matrices:\n'
     '\n\t{0}\n\t{1}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         XYZ, XYZ_0)))
print(colour.whiteness(XYZ=XYZ, XYZ_0=XYZ_0, method='Berger 1959'))
print(colour.colorimetry.whiteness_Berger1959(XYZ, XYZ_0))

print('\n')

message_box(('Computing "whiteness" using "Taube (1960)" method for '
             'given sample and reference white "CIE XYZ" tristimulus values '
             'matrices:\n'
             '\n\t{0}\n\t{1}'.format(XYZ, XYZ_0)))
print(colour.whiteness(XYZ=XYZ, XYZ_0=XYZ_0, method='Taube 1960'))
print(colour.colorimetry.whiteness_Taube1960(XYZ, XYZ_0))

print('\n')

Lab = np.array([100.00000000, -2.46875131, -16.72486654])
message_box(('Computing "whiteness" using "Stensby (1968)" method for '
             'given sample "CIE L*a*b*" colourspace array:\n'
             '\n\t{0}'.format(Lab)))
print(colour.whiteness(Lab=Lab, method='Stensby 1968'))
print(colour.colorimetry.whiteness_Stensby1968(Lab))

print('\n')

message_box(('Computing "whiteness" using "ASTM E313" method for '
             'given sample "CIE XYZ" tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.whiteness(XYZ=XYZ, method='ASTM E313'))
print(colour.colorimetry.whiteness_ASTME313(XYZ))

print('\n')

xy = np.array([0.3167, 0.3334])
Y = 100
message_box(
    ('Computing "whiteness" using "Ganz and Griesser (1979)" method '
     'for given sample "xy" chromaticity coordinates, "Y" tristimulus '
     'value:\n'
     '\n\t{0}\n\t{1}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         xy, Y)))
print(colour.whiteness(xy=xy, Y=Y, method='Ganz 1979'))
print(colour.colorimetry.whiteness_Ganz1979(xy, Y))

print('\n')

xy = np.array([0.3167, 0.3334])
Y = 100
xy_n = np.array([0.3139, 0.3311])
message_box(
    ('Computing "whiteness" using "CIE 2004" method for '
     'given sample "xy" chromaticity coordinates, "Y" tristimulus '
     'value and "xy_n" chromaticity coordinates of perfect diffuser:\n'
     '\n\t{0}\n\t{1}\n\t{2}\n\n'
     'Warning: The input domain of that definition is non standard!'.format(
         xy, Y, xy_n)))
print(colour.whiteness(xy=xy, Y=Y, xy_n=xy_n))
print(colour.colorimetry.whiteness_CIE2004(xy, Y, xy_n))
