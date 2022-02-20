"""Showcases colour models computations."""

import numpy as np

import colour
from colour.utilities import message_box

message_box("Colour Models Computations")

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(
    f'Converting to the "CIE xyY" colourspace from given "CIE XYZ" '
    f"tristimulus values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_xyY(XYZ))

print("\n")

message_box(
    "The default illuminant if X == Y == Z == 0 is "
    '"CIE Standard Illuminant D Series D65".'
)
print(colour.XYZ_to_xyY(np.array([0.00000000, 0.00000000, 0.00000000])))

print("\n")

message_box("Using an alternative illuminant.")
print(
    colour.XYZ_to_xyY(
        np.array([0.00000000, 0.00000000, 0.00000000]),
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["ACES"],
    )
)

print("\n")

xyY = np.array([0.26414772, 0.37770001, 0.10080000])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "CIE xyY" '
    f"colourspace values:\n\n\t{xyY}"
)
print(colour.xyY_to_XYZ(xyY))

print("\n")

message_box(
    f'Converting to "xy" chromaticity coordinates from given "CIE XYZ" '
    f"tristimulus values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_xy(XYZ))

print("\n")

xy = np.array([0.26414772, 0.37770001])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "xy" chromaticity '
    f"coordinates:\n\n\t{xy}"
)
print(colour.xy_to_XYZ(xy))

print("\n")

message_box(
    f'Converting to "RGB" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
print(
    colour.XYZ_to_RGB(
        XYZ,
        D65,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        "Bradford",
        colour.RGB_COLOURSPACES["sRGB"].cctf_encoding,
    )
)

print("\n")

RGB = np.array([0.45620519, 0.03081071, 0.04091952])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "RGB" colourspace '
    f"values:\n\n\t{RGB}"
)
print(
    colour.RGB_to_XYZ(
        RGB,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        D65,
        colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ,
        "Bradford",
        colour.RGB_COLOURSPACES["sRGB"].cctf_decoding,
    )
)

print("\n")

message_box(
    f'Converting to "sRGB" colourspace from given "CIE XYZ" tristimulus '
    f"values using convenient definition:\n\n\t{XYZ}"
)
print(colour.XYZ_to_sRGB(XYZ, D65))

print("\n")

message_box(
    f'Converting to "CIE 1960 UCS" colourspace from given "CIE XYZ" '
    f"tristimulus values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_UCS(XYZ))

print("\n")

UCS = np.array([0.07049533, 0.10080000, 0.09558313])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "CIE 1960 UCS" '
    f"colourspace values:\n\n\t{UCS}"
)
print(colour.UCS_to_XYZ(UCS))

print("\n")

message_box(
    f'Converting to "uv" chromaticity coordinates from given "CIE UCS" '
    f"colourspace values:\n\n\t{UCS}"
)
print(colour.UCS_to_uv(UCS))

print("\n")

uv = np.array([0.15085309, 0.32355314])
message_box(
    f'Converting to "xy" chromaticity coordinates from given "CIE UCS" '
    f'colourspace "uv" chromaticity coordinates:\n\n\t{uv}'
)
print(colour.UCS_uv_to_xy(uv))

print("\n")

xy = np.array([0.26414771, 0.37770001])
message_box(
    f'Converting to "CIE UCS" colourspace "uv" chromaticity coordinates from '
    f'given "xy" chromaticity coordinates:\n\n\t{xy}'
)
print(colour.xy_to_UCS_uv(xy))

print("\n")

message_box(
    f'Converting to "CIE 1964 U*V*W*" colourspace from given"CIE XYZ" '
    f"tristimulus values:\n\n\t{XYZ * 100}"
)
print(colour.XYZ_to_UVW(XYZ * 100))

print("\n")

UVW = np.array([-22.59840563, 5.45505477, 37.00411491])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given"CIE 1964 U*V*W*" '
    f"colourspace values:\n\n\t{UVW}"
)
print(colour.UVW_to_XYZ(UVW) / 100)

print("\n")

message_box(
    f'Converting to "CIE L*u*v*" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_Luv(XYZ))

print("\n")

Luv = np.array([37.9856291, -23.19781615, 8.39962073])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "CIE L*u*v*" '
    f"colourspace values:\n\n\t{Luv}"
)
print(colour.Luv_to_XYZ(Luv))

print("\n")

message_box(
    f'Converting to "u"v"" chromaticity coordinates from given "CIE L*u*v*" '
    f"colourspace values:\n\n\t{Luv}"
)
print(colour.Luv_to_uv(Luv))

print("\n")

uv = np.array([0.1508531, 0.48532971])
message_box(
    f'Converting to "xy" chromaticity coordinates from given "CIE L*u*v*" '
    f'colourspace "u"v"" chromaticity coordinates:\n\n\t{uv}'
)
print(colour.Luv_uv_to_xy(uv))

print("\n")

xy = np.array([0.26414771, 0.37770001])
message_box(
    f'Converting to "CIE L*u*v*" colourspace "u"v"" chromaticity coordinates '
    f'from given "xy" chromaticity coordinates:\n\n\t{xy}'
)
print(colour.xy_to_Luv_uv(xy))

print("\n")

message_box(
    f'Converting to "CIE L*C*Huv" colourspace from given "CIE L*u*v*" '
    f"colourspace values:\n\n\t{Luv}"
)
print(colour.Luv_to_LCHuv(Luv))

print("\n")

LCHuv = np.array([37.9856291, 24.67169031, 160.09535205])
message_box(
    f'Converting to "CIE L*u*v*" colourspace from given "CIE L*C*Huv" '
    f"colourspace values:\n\n\t{LCHuv}"
)
print(colour.LCHuv_to_Luv(LCHuv))

print("\n")

message_box(
    f'Converting to "CIE L*a*b*" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_Lab(XYZ))

print("\n")

Lab = np.array([37.9856291, -22.61920654, 4.19811236])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "CIE L*a*b*" '
    f"colourspace values:\n\n\t{Lab}"
)
print(colour.Lab_to_XYZ(Lab))

print("\n")

message_box(
    f'Converting to "CIE L*C*Hab" colourspace from given "CIE L*a*b*" '
    f"colourspace values:\n\n\t{Lab}"
)
print(colour.Lab_to_LCHab(Lab))

print("\n")

LCHab = np.array([37.9856291, 23.00549178, 169.48557589])
message_box(
    f'Converting to "CIE L*a*b*" colourspace from given "CIE L*C*Hab" '
    f"colourspace values:\n\n\t{LCHab}"
)
print(colour.LCHab_to_Lab(LCHab))

print("\n")

XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
message_box(
    f'Converting to "Hunter L,a,b" colour scale from given "CIE XYZ" '
    f"tristimulus values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_Hunter_Lab(XYZ))

print("\n")

Lab = np.array([31.74901573, -14.44108591, 2.74396261])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "Hunter L,a,b" '
    f"colour scale values:\n\n\t{Lab}"
)
print(colour.Hunter_Lab_to_XYZ(Lab))

print("\n")

message_box(
    f'Converting to "Hunter Rd,a,b" colour scale from given "CIE XYZ" '
    f"tristimulus values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_Hunter_Rdab(XYZ))

print("\n")

R_d_ab = np.array([10.08000000, -17.8442708, 3.39060457])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given"Hunter Rd,a,b" '
    f"colour scale values:\n\n\t{R_d_ab}"
)
print(colour.Hunter_Rdab_to_XYZ(R_d_ab))

print("\n")

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(
    f'Converting to "ICaCb" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_ICaCb(XYZ))

ICaCb = np.array([0.06875297, 0.05753352, 0.02081548])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "ICaCb" '
    f"colourspace values:\n\n\t{ICaCb}"
)
print(colour.ICaCb_to_XYZ(ICaCb))

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(
    f'Converting to "IgPgTg" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_IgPgTg(XYZ))

print("\n")

IgPgTg = np.array([0.42421258, 0.18632491, 0.10689223])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "IgPgTg" '
    f"colourspace values:\n\n\t{IgPgTg}"
)
print(colour.IgPgTg_to_XYZ(IgPgTg))

print("\n")

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(
    f'Converting to "IPT" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_IPT(XYZ))

print("\n")

IPT = np.array([0.36571124, -0.11114798, 0.01594746])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "IPT" colourspace '
    f"values:\n\n\t{IPT}"
)
print(colour.IPT_to_XYZ(IPT))

print("\n")

message_box(
    f'Converting to "hdr-CIELab" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_hdr_CIELab(XYZ))

print("\n")

Lab_hdr = np.array([48.26598942, -26.97517728, 4.99243377])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "hdr-CIELab" '
    f"colourspace values:\n\n\t{Lab_hdr}"
)
print(colour.hdr_CIELab_to_XYZ(Lab_hdr))

print("\n")

message_box(
    f'Converting to "hdr-IPT" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_hdr_IPT(XYZ))

print("\n")

IPT_hdr = np.array([46.4993815, -12.82251566, 1.85029518])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "hdr-IPT" '
    f"colourspace values:\n\n\t{IPT_hdr}"
)
print(colour.hdr_IPT_to_XYZ(IPT_hdr))

print("\n")

message_box(
    f'Converting to "Jzazbz" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_Jzazbz(XYZ))

print("\n")

Jzazbz = np.array([0.00357804, -0.00295507, 0.00038998])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from given "Jzazbz" '
    f"colourspace values:\n\n\t{Jzazbz}"
)
print(colour.Jzazbz_to_XYZ(Jzazbz))

print("\n")

message_box(
    f'Converting to "OSA UCS" colourspace from given "CIE XYZ" tristimulus '
    f'values under the "CIE 1964 10 Degree Standard Observer":\n\n'
    f"\t{XYZ * 100}"
)
print(colour.XYZ_to_OSA_UCS(XYZ * 100))

print("\n")

Ljg = np.array([-4.4900683, 0.70305936, 3.03463664])
message_box(
    f'Converting to "CIE XYZ" tristimulus values under the '
    f'"CIE 1964 10 Degree Standard Observer" from "OSA UCS" colourspace:\n\n'
    f"\t{Ljg}"
)
print(colour.OSA_UCS_to_XYZ(Ljg))

print("\n")

message_box(
    f'Converting to "Oklab" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{XYZ}"
)
print(colour.XYZ_to_Oklab(XYZ))

print("\n")

Lab = np.array([0.51634019, 0.15469500, 0.06289579])
message_box(
    f'Converting to "CIE XYZ" tristimulus values from "Oklab" colourspace:\n\n'
    f"\t{Lab}"
)
print(colour.Oklab_to_XYZ(Lab))

print("\n")

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box(
    'Converting to the "ProLab" colourspace from given "CIE XYZ" '
    "tristimulus values:\n\n"
    f"\t{XYZ}"
)
print(colour.XYZ_to_ProLab(XYZ))

print("\n")

ProLab = np.array([0.36571124, -0.11114798, 0.01594746])
message_box(
    'Converting to "CIE XYZ" tristimulus values from given "ProLab" '
    "colourspace values:\n\n"
    f"\t{ProLab}"
)
print(colour.ProLab_to_XYZ(ProLab))

print("\n")

XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]
specification = colour.XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
JMh = (specification.J, specification.M, specification.h)
message_box(
    f'Converting to "CAM02-UCS" colourspace from given "CIECAM02" colour '
    f'appearance model "JMh" correlates:\n\n\t{JMh}'
)
print(colour.JMh_CIECAM02_to_CAM02UCS(JMh))

print("\n")

message_box(
    f'Converting to "CAM02-UCS" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{JMh}"
)
print(colour.XYZ_to_CAM02UCS(XYZ / 100, XYZ_w=XYZ_w / 100, L_A=L_A, Y_b=Y_b))

print("\n")

specification = colour.XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)
JMh = (specification.J, specification.M, specification.h)
message_box(
    f'Converting to "CAM16-UCS" colourspace from given "CAM16" colour '
    f'appearance model "JMh" correlates:\n\n\t{JMh}'
)
print(colour.JMh_CAM16_to_CAM16UCS(JMh))

message_box(
    f'Converting to "CAM16-UCS" colourspace from given "CIE XYZ" tristimulus '
    f"values:\n\n\t{JMh}"
)
print(colour.XYZ_to_CAM16UCS(XYZ / 100, XYZ_w=XYZ_w / 100, L_A=L_A, Y_b=Y_b))
