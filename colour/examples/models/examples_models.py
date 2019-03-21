# -*- coding: utf-8 -*-
"""
Showcases colour models computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('Colour Models Computations')

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(('Converting to "CIE xyY" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_xyY(XYZ))

print('\n')

message_box(('The default illuminant if X == Y == Z == 0 is '
             '"CIE Standard Illuminant D Series D65".'))
print(colour.XYZ_to_xyY(np.array([0.00000000, 0.00000000, 0.00000000])))

print('\n')

message_box('Using an alternative illuminant.')
print(
    colour.XYZ_to_xyY(
        np.array([0.00000000, 0.00000000, 0.00000000]),
        colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['ACES'],
    ))

print('\n')

xyY = np.array([0.26414772, 0.37770001, 0.10080000])
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

xy = np.array([0.26414772, 0.37770001])
message_box(('Converting to "CIE XYZ" tristimulus values from given "xy" '
             'chromaticity coordinates:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_XYZ(xy))

print('\n')

message_box(('Converting to "RGB" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
D65 = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
print(
    colour.XYZ_to_RGB(
        XYZ,
        D65,
        colour.RGB_COLOURSPACES['sRGB'].whitepoint,
        colour.RGB_COLOURSPACES['sRGB'].XYZ_to_RGB_matrix,
        'Bradford',
        colour.RGB_COLOURSPACES['sRGB'].encoding_cctf,
    ))

print('\n')

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(('Converting to "CIE XYZ" tristimulus values from given "RGB" '
             'colourspace values:\n'
             '\n\t{0}'.format(RGB)))
print(
    colour.RGB_to_XYZ(
        RGB,
        colour.RGB_COLOURSPACES['sRGB'].whitepoint,
        D65,
        colour.RGB_COLOURSPACES['sRGB'].RGB_to_XYZ_matrix,
        'Bradford',
        colour.RGB_COLOURSPACES['sRGB'].decoding_cctf,
    ))

print('\n')

message_box(('Converting to "sRGB" colourspace from given "CIE XYZ" '
             'tristimulus values using convenient definition:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_sRGB(XYZ, D65))

print('\n')

message_box(('Converting to "CIE 1960 UCS" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_UCS(XYZ))

print('\n')

UCS = np.array([0.07049533, 0.10080000, 0.09558313])
message_box(('Converting to "CIE XYZ" tristimulus values from given'
             '"CIE 1960 UCS" colourspace values:\n'
             '\n\t{0}'.format(UCS)))
print(colour.UCS_to_XYZ(UCS))

print('\n')

message_box(('Converting to "uv" chromaticity coordinates from given '
             '"CIE UCS" colourspace values:\n'
             '\n\t{0}'.format(UCS)))
print(colour.UCS_to_uv(UCS))

print('\n')

uv = np.array([0.15085309, 0.32355314])
message_box(('Converting to "xy" chromaticity coordinates from given '
             '"CIE UCS" colourspace "uv" chromaticity coordinates:\n'
             '\n\t{0}'.format(uv)))
print(colour.UCS_uv_to_xy(uv))

print('\n')

xy = np.array([0.26414771, 0.37770001])
message_box(('Converting to "CIE UCS" colourspace "uv" chromaticity '
             'coordinates from given "xy" chromaticity coordinates:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_UCS_uv(xy))

print('\n')

message_box(('Converting to "CIE 1964 U*V*W*" colourspace from given'
             '"CIE XYZ" tristimulus values:\n'
             '\n\t{0}'.format(XYZ * 100)))
print(colour.XYZ_to_UVW(XYZ * 100))

print('\n')

UVW = np.array([-22.59840563, 5.45505477, 37.00411491])
message_box(('Converting to "CIE XYZ" tristimulus values from given'
             '"CIE 1964 U*V*W*" colourspace values:\n'
             '\n\t{0}'.format(UVW)))
print(colour.UVW_to_XYZ(UVW) / 100)

print('\n')

message_box(('Converting to "CIE L*u*v*" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_Luv(XYZ))

print('\n')

Luv = np.array([37.9856291, -23.19781615, 8.39962073])
message_box(('Converting to "CIE XYZ" tristimulus values from given '
             '"CIE L*u*v*" colourspace values:\n'
             '\n\t{0}'.format(Luv)))
print(colour.Luv_to_XYZ(Luv))

print('\n')

message_box(('Converting to "u"v"" chromaticity coordinates from given '
             '"CIE L*u*v*" colourspace values:\n'
             '\n\t{0}'.format(Luv)))
print(colour.Luv_to_uv(Luv))

print('\n')

uv = np.array([0.1508531, 0.48532971])
message_box(('Converting to "xy" chromaticity coordinates from given '
             '"CIE L*u*v*" colourspace "u"v"" chromaticity coordinates:\n'
             '\n\t{0}'.format(uv)))
print(colour.Luv_uv_to_xy(uv))

print('\n')

xy = np.array([0.26414771, 0.37770001])
message_box(('Converting to "CIE L*u*v*" colourspace "u"v"" chromaticity '
             'coordinates from given "xy" chromaticity coordinates:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_Luv_uv(xy))

print('\n')

message_box(('Converting to "CIE L*C*Huv" colourspace from given "CIE L*u*v*" '
             'colourspace values:\n'
             '\n\t{0}'.format(Luv)))
print(colour.Luv_to_LCHuv(Luv))

print('\n')

LCHuv = np.array([37.9856291, 24.67169031, 160.09535205])
message_box(('Converting to "CIE L*u*v*" colourspace from given "CIE L*C*Huv" '
             'colourspace values:\n'
             '\n\t{0}'.format(LCHuv)))
print(colour.LCHuv_to_Luv(LCHuv))

print('\n')

message_box(('Converting to "CIE L*a*b*" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_Lab(XYZ))

print('\n')

Lab = np.array([37.9856291, -22.61920654, 4.19811236])
message_box(('Converting to "CIE XYZ" tristimulus values from given '
             '"CIE L*a*b*" colourspace values:\n'
             '\n\t{0}'.format(Lab)))
print(colour.Lab_to_XYZ(Lab))

print('\n')

message_box(('Converting to "CIE L*C*Hab" colourspace from given "CIE L*a*b*" '
             'colourspace values:\n'
             '\n\t{0}'.format(Lab)))
print(colour.Lab_to_LCHab(Lab))

print('\n')

LCHab = np.array([37.9856291, 23.00549178, 169.48557589])
message_box(('Converting to "CIE L*a*b*" colourspace from given "CIE L*C*Hab" '
             'colourspace values:\n'
             '\n\t{0}'.format(LCHab)))
print(colour.LCHab_to_Lab(LCHab))

print('\n')

XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
message_box(('Converting to "Hunter L,a,b" colour scale from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_Hunter_Lab(XYZ))

print('\n')

Lab = np.array([31.74901573, -14.44108591, 2.74396261])
message_box(('Converting to "CIE XYZ" tristimulus values from given '
             '"Hunter L,a,b" colour scale values:\n'
             '\n\t{0}'.format(Lab)))
print(colour.Hunter_Lab_to_XYZ(Lab))

print('\n')

message_box(('Converting to "Hunter Rd,a,b" colour scale from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_Hunter_Rdab(XYZ))

print('\n')

R_d_ab = np.array([10.08000000, -17.8442708, 3.39060457])
message_box(('Converting to "CIE XYZ" tristimulus values from given'
             '"Hunter Rd,a,b" colour scale values:\n'
             '\n\t{0}'.format(R_d_ab)))
print(colour.Hunter_Rdab_to_XYZ(R_d_ab))

print('\n')

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(('Converting to "IPT" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_IPT(XYZ))

print('\n')

IPT = np.array([0.36571124, -0.11114798, 0.01594746])
message_box(('Converting to "CIE XYZ" tristimulus values from given "IPT" '
             'colourspace values:\n'
             '\n\t{0}'.format(IPT)))
print(colour.IPT_to_XYZ(IPT))

print('\n')

message_box(('Converting to "hdr-CIELab" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_hdr_CIELab(XYZ))

print('\n')

Lab_hdr = np.array([48.26598942, -26.97517728, 4.99243377])
message_box(('Converting to "CIE XYZ" tristimulus values from given '
             '"hdr-CIELab" colourspace values:\n'
             '\n\t{0}'.format(Lab_hdr)))
print(colour.hdr_CIELab_to_XYZ(Lab_hdr))

print('\n')

message_box(('Converting to "hdr-IPT" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_hdr_IPT(XYZ))

print('\n')

IPT_hdr = np.array([46.4993815, -12.82251566, 1.85029518])
message_box(('Converting to "CIE XYZ" tristimulus values from given "hdr-IPT" '
             'colourspace values:\n'
             '\n\t{0}'.format(IPT_hdr)))
print(colour.hdr_IPT_to_XYZ(IPT_hdr))

print('\n')

message_box(('Converting to "JzAzBz" colourspace from given "CIE XYZ" '
             'tristimulus values:\n'
             '\n\t{0}'.format(XYZ)))
print(colour.XYZ_to_JzAzBz(XYZ))

print('\n')

JzAzBz = np.array([0.00357804, -0.00295507, 0.00038998])
message_box(('Converting to "CIE XYZ" tristimulus values from given "JzAzBz" '
             'colourspace values:\n'
             '\n\t{0}'.format(JzAzBz)))
print(colour.JzAzBz_to_XYZ(JzAzBz))

print('\n')

message_box(('Converting to "OSA UCS" colourspace from given "CIE XYZ" '
             'tristimulus values under the '
             '"CIE 1964 10 Degree Standard Observer":\n'
             '\n\t{0}'.format(XYZ * 100)))
print(colour.XYZ_to_OSA_UCS(XYZ * 100))

print('\n')

Ljg = np.array([-4.4900683, 0.70305936, 3.03463664])
message_box(('Converting to "CIE XYZ" tristimulus values under the '
             '"CIE 1964 10 Degree Standard Observer" '
             'from "OSA UCS" colourspace:\n'
             '\n\t{0}'.format(Ljg)))
print(colour.OSA_UCS_to_XYZ(Ljg))

print('\n')

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31
Y_b = 20.0
surround = colour.CIECAM02_VIEWING_CONDITIONS['Average']
specification = colour.XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
JMh = (specification.J, specification.M, specification.h)
message_box(('Converting to "CAM02-UCS" colourspace from given '
             '"CIECAM02" colour appearance model "JMh" correlates:\n'
             '\n\t{0}'.format(JMh)))
print(colour.JMh_CIECAM02_to_CAM02UCS(JMh))

print('\n')

specification = colour.XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)
JMh = (specification.J, specification.M, specification.h)
message_box(('Converting to "CAM16-UCS" colourspace from given '
             '"CAM16" colour appearance model "JMh" correlates:\n'
             '\n\t{0}'.format(JMh)))
print(colour.JMh_CAM16_to_CAM16UCS(JMh))
