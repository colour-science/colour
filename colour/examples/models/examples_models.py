#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour models computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('Colour Models Computations')

XYZ = (1.14176346, 1.00000000, 0.49815206)
message_box(('Converting to "CIE xyY" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_xyY(XYZ))

print('\n')

message_box(('The default illuminant if X == Y == Z == 0 is '
             'CIE Standard Illuminant D Series D50".'))
print(colour.XYZ_to_xyY((0.00000000, 0.00000000, 0.00000000)))

print('\n')

message_box('Using an alternative illuminant.')
print(colour.XYZ_to_xyY(
    (0.00000000, 0.00000000, 0.00000000),
    colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D60']))

print('\n')

xyY = (0.4325, 0.3788, 1.0000)
message_box(('Converting to "CIE XYZ" tristimulus values from given "CIE xyY" '
             'colourspace values:\n'
             '\n\t{0}'.format(xyY)))
print(colour.xyY_to_XYZ(xyY))

print('\n')

message_box(('Converting to "xy" chromaticity coordinates from given '
             '"CIE XYZ" tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_xy(XYZ))

print('\n')

xy = (0.43249999995420696, 0.378800000065942)
message_box(('Converting to "CIE XYZ" tristimulus values from given "xy" '
             'chromaticity coordinates:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_XYZ(xy))

print('\n')

message_box(('Converting to "RGB" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_RGB(
    XYZ,
    colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
    colour.sRGB_COLOURSPACE.whitepoint,
    colour.sRGB_COLOURSPACE.XYZ_to_RGB_matrix,
    'Bradford',
    colour.sRGB_COLOURSPACE.OECF))

print('\n')

RGB = (1.26651054, 0.91394181, 0.76936593)
message_box(('Converting to "CIE XYZ" tristimulus values from given "RGB" '
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(colour.RGB_to_XYZ(
    RGB,
    colour.sRGB_COLOURSPACE.whitepoint,
    colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
    colour.sRGB_COLOURSPACE.RGB_to_XYZ_matrix,
    'Bradford',
    colour.sRGB_COLOURSPACE.EOCF))

print('\n')

message_box(('Converting to "sRGB" colourspace from given "CIE XYZ" '
             'tristimulus values using convenient definition:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_sRGB(
    XYZ,
    colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']))

print('\n')

message_box(('Converting to "CIE UCS" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_UCS(XYZ))

print('\n')

UCS = (0.76117564, 1.00000000, 1.17819430)
message_box(('Converting to "CIE XYZ" tristimulus values from given "CIE UCS" '
             'colourspace values:\n'
             '\n\t{0}'.format(UCS)))
print(colour.UCS_to_XYZ(UCS))

print('\n')

message_box(('Converting to "uv" chromaticity coordinates from given '
             '"CIE UCS" colourspace values:\n'
             '\n\t{0}'.format(UCS)))
print(colour.UCS_to_uv(UCS))

print('\n')

uv = (0.25895877536258677, 0.34020896328551287)
message_box(('Converting to "xy" chromaticity coordinates from given '
             '"CIE UCS" colourspace "uv" chromaticity coordinates:\n'
             '\n\t{0}'.format(uv)))
print(colour.UCS_uv_to_xy(uv))

print('\n')

message_box(('Converting to "CIE UVW" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_UVW(XYZ))

print('\n')

message_box(('Converting to "CIE Luv" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_Luv(XYZ))

print('\n')

Luv = (100.00000000, 64.73951819, 28.90956141)
message_box(('Converting to "CIE XYZ" tristimulus values from given "CIE Luv" '
             'colourspace values:\n'
             '\n\t{0}'.format(Luv)))
print(colour.Luv_to_XYZ(Luv))

print('\n')

message_box(('Converting to "u"v"" chromaticity coordinates from given '
             '"CIE Luv" colourspace values:\n'
             '\n\t{0}'.format(Luv)))
print(colour.Luv_to_uv(Luv))

print('\n')

uv = (0.25895877536234663, 0.51031344493132547)
message_box(('Converting to "xy" chromaticity coordinates from given '
             '"CIE Luv" colourspace "u"v"" chromaticity coordinates:\n'
             '\n\t{0}'.format(uv)))
print(colour.Luv_uv_to_xy(uv))

print('\n')

message_box(('Converting to "CIE LCHuv" colourspace from given "CIE Luv" '
             'colourspace values:\n'
             '\n\t{0}'.format(Luv)))
print(colour.Luv_to_LCHuv(Luv))

print('\n')

LCHuv = (100.00000000, 70.90111393, 24.06324597)
message_box(('Converting to "CIE Luv" colourspace from given "CIE LCHuv" '
             'colourspace values:\n'
             '\n\t{0}'.format(LCHuv)))
print(colour.LCHuv_to_Luv(LCHuv))

print('\n')

message_box(('Converting to "CIE Lab" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_Lab(XYZ))

print('\n')

Lab = (100.00000000, 28.97832184, 30.96902832)
message_box(('Converting to "CIE XYZ" tristimulus values from given "CIE Lab" '
             'colourspace values:\n'
             '\n\t{0}'.format(Lab)))
print(colour.Lab_to_XYZ(Lab))

print('\n')

message_box(('Converting to "CIE LCHab" colourspace from given "CIE Lab" '
             'colourspace values:\n'
             '\n\t{0}'.format(Lab)))
print(colour.Lab_to_LCHab(Lab))

print('\n')

LCHab = (100.00000000, 42.41254357, 46.90195532)
message_box(('Converting to "CIE Lab" colourspace from given "CIE LCHab" '
             'colourspace values:\n'
             '\n\t{0}'.format(LCHab)))
print(colour.LCHab_to_Lab(LCHab))
