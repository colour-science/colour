"""Showcases *whiteness* computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Whiteness" Computations')

XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
message_box(
    f'Computing "whiteness" using "Berger (1959)" method for given sample and '
    f'reference white "CIE XYZ" tristimulus values matrices:\n\n'
    f"\t{XYZ}\n"
    f"\t{XYZ_0}"
)
print(colour.whiteness(XYZ, XYZ_0, method="Berger 1959"))
print(colour.colorimetry.whiteness_Berger1959(XYZ, XYZ_0))

print("\n")

message_box(
    f'Computing "whiteness" using "Taube (1960)" method for given sample and '
    f'reference white "CIE XYZ" tristimulus values matrices:\n\n'
    f"\t{XYZ}\n"
    f"\t{XYZ_0}"
)
print(colour.whiteness(XYZ, XYZ_0, method="Taube 1960"))
print(colour.colorimetry.whiteness_Taube1960(XYZ, XYZ_0))

print("\n")

Lab = colour.XYZ_to_Lab(XYZ / 100, colour.XYZ_to_xy(XYZ_0 / 100))
message_box(
    f'Computing "whiteness" using "Stensby (1968)" method for given sample '
    f'"CIE L*a*b*" colourspace array:\n\n\t{Lab}'
)
print(colour.whiteness(XYZ, XYZ_0, method="Stensby 1968"))
print(colour.colorimetry.whiteness_Stensby1968(Lab))

print("\n")

message_box(
    f'Computing "whiteness" using "ASTM E313" method for given sample '
    f'"CIE XYZ" tristimulus values:\n\n\t{XYZ}'
)
print(colour.whiteness(XYZ, XYZ_0, method="ASTM E313"))
print(colour.colorimetry.whiteness_ASTME313(XYZ))

print("\n")

xy = colour.XYZ_to_xy(XYZ / 100)
Y = 100
message_box(
    f'Computing "whiteness" using "Ganz and Griesser (1979)" method for given '
    f'sample "xy" chromaticity coordinates, "Y" tristimulus value:\n\n'
    f"\t{xy}\n"
    f"\t{Y}"
)
print(colour.whiteness(XYZ, XYZ_0, method="Ganz 1979"))
print(colour.colorimetry.whiteness_Ganz1979(xy, Y))

print("\n")

xy = colour.XYZ_to_xy(XYZ / 100)
Y = 100
xy_n = colour.XYZ_to_xy(XYZ_0 / 100)
message_box(
    f'Computing "whiteness" using "CIE 2004" method for given sample "xy" '
    f'chromaticity coordinates, "Y" tristimulus value and "xy_n" chromaticity '
    f"coordinates of perfect diffuser:\n\n"
    f"\t{xy}\n"
    f"\t{Y}\n"
    f"\t{xy_n}"
)
print(colour.whiteness(XYZ, XYZ_0, xy_n=xy_n))
print(colour.colorimetry.whiteness_CIE2004(xy, Y, xy_n))
