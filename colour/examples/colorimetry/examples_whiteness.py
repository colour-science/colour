#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *whiteness* computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"Whiteness" Computations')

XYZ = (95.00000000, 100.00000000, 105.00000000)
XYZ_0 = (94.80966767, 100.00000000, 107.30513595)
message_box(('Computing "whiteness" using "Berger (1959)" method for '
             'given sample and reference white "CIE XYZ" tristimulus values '
             'matrices:\n'
             '\n\t{0}\n\t{1}'.format(XYZ, XYZ_0)))
print(colour.whiteness_Berger1959(XYZ, XYZ_0))
print(colour.whiteness(XYZ=XYZ, XYZ_0=XYZ_0, method='Berger 1959'))

print('\n')

message_box(('Computing "whiteness" using "Taube (1960)" method for '
             'given sample and reference white "CIE XYZ" tristimulus values '
             'matrices:\n'
             '\n\t{0}\n\t{1}'.format(XYZ, XYZ_0)))
print(colour.whiteness_Taube1960(XYZ, XYZ_0))
print(colour.whiteness(XYZ=XYZ, XYZ_0=XYZ_0, method='Taube 1960'))

print('\n')

Lab = (100.00000000, -2.46875131, -16.72486654)
message_box(('Computing "whiteness" using "Stensby (1968)" method for '
             'given sample "CIE Lab" colourspace array:\n'
             '\n\t{0}'.format(Lab)))
print(colour.whiteness_Stensby1968(Lab))
print(colour.whiteness(Lab=Lab, method='Stensby 1968'))

print('\n')

message_box(('Computing "whiteness" using "ASTM 313" method for '
             'given sample "CIE XYZ" tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.whiteness_ASTM313(XYZ))
print(colour.whiteness(XYZ=XYZ, method='ASTM 313'))

print('\n')

xy = (0.3167, 0.3334)
Y = 100
message_box(('Computing "whiteness" using "Ganz and Griesser (1979)" method '
             'for given sample "xy" chromaticity coordinates, "Y" tristimulus '
             'value:\n'
             '\n\t{0}\n\t{1}'.format(xy, Y)))
print(colour.whiteness_Ganz1979(xy, Y))
print(colour.whiteness(xy=xy, Y=Y, method='Ganz 1979'))

print('\n')

xy = (0.3167, 0.3334)
Y = 100
xy_n = (0.3139, 0.3311)
message_box(('Computing "whiteness" using "CIE 2004" method for '
             'given sample "xy" chromaticity coordinates, "Y" tristimulus '
             'value and "xy_n" chromaticity coordinates of perfect diffuser:\n'
             '\n\t{0}\n\t{1}\n\t{2}'.format(xy, Y, xy_n)))
print(colour.whiteness_CIE2004(xy, Y, xy_n))
print(colour.whiteness(xy=xy, Y=Y, xy_n=xy_n))
