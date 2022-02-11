"""Showcases correlated colour temperature computations."""

import colour
from colour.utilities import message_box

message_box("Correlated Colour Temperature Computations")

cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
illuminant = colour.SDS_ILLUMINANTS["D65"]
xy = colour.XYZ_to_xy(colour.sd_to_XYZ(illuminant, cmfs) / 100)
uv = colour.UCS_to_uv(colour.XYZ_to_UCS(colour.xy_to_XYZ(xy)))
message_box(
    f'Converting to "CCT" and "D_uv" from given "CIE UCS" colourspace "uv" '
    f'chromaticity coordinates using "Ohno (2013)" method:\n\n\t{uv}'
)
print(colour.uv_to_CCT(uv, cmfs=cmfs))
print(colour.temperature.uv_to_CCT_Ohno2013(uv, cmfs=cmfs))

print("\n")

message_box("Faster computation with 3 iterations but a lot less precise.")
print(colour.uv_to_CCT(uv, cmfs=cmfs, iterations=3))
print(colour.temperature.uv_to_CCT_Ohno2013(uv, cmfs=cmfs, iterations=3))

print("\n")

message_box(
    f'Converting to "CCT" and "D_uv" from given "CIE UCS" colourspace "uv" '
    f'chromaticity coordinates using "Robertson (1968)" method:\n\n\t{uv}'
)
print(colour.uv_to_CCT(uv, method="Robertson 1968"))
print(colour.temperature.uv_to_CCT_Robertson1968(uv))

print("\n")

CCT_D_uv = [6503.49254150, 0.00320598]
message_box(
    f'Converting to "CIE UCS" colourspace "uv" chromaticity coordinates from '
    f'given "CCT" and "D_uv" using "Ohno (2013)" method:\n\n\t{CCT_D_uv}'
)
print(colour.CCT_to_uv(CCT_D_uv, cmfs=cmfs))
print(colour.temperature.CCT_to_uv_Ohno2013(CCT_D_uv, cmfs=cmfs))

print("\n")

message_box(
    f'Converting to "CIE UCS" colourspace "uv" chromaticity coordinates from '
    f'given "CCT" and "D_uv" using "Robertson (1968)" method:\n\n\t{CCT_D_uv}'
)
print(colour.CCT_to_uv(CCT_D_uv, method="Robertson 1968"))
print(colour.temperature.CCT_to_uv_Robertson1968(CCT_D_uv))

print("\n")

CCT = 6503.49254150
message_box(
    f'Converting to "CIE UCS" colourspace "uv" chromaticity coordinates from '
    f'given "CCT" using "Krystek (1985)" method:\n\n\t({CCT})'
)
print(colour.CCT_to_uv(CCT, method="Krystek 1985"))
print(colour.temperature.CCT_to_uv_Krystek1985(CCT))

print("\n")

xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
message_box(
    f'Converting to "CCT" from given "CIE xy" chromaticity coordinates using '
    f'"McCamy (1992)" method:\n\n\t{xy}'
)
print(colour.xy_to_CCT(xy, method="McCamy 1992"))
print(colour.temperature.xy_to_CCT_McCamy1992(xy))

print("\n")

message_box(
    f'Converting to "CCT" from given "CIE xy" chromaticity coordinates using '
    f'"Hernandez-Andres, Lee and Romero (1999)" method:\n\n\t{xy}'
)
print(colour.xy_to_CCT(xy, method="Hernandez 1999"))
print(colour.temperature.xy_to_CCT_Hernandez1999(xy))

print("\n")

CCT = 6503.49254150
message_box(
    f'Converting to "CIE xy" chromaticity coordinates from given "CCT" using '
    f'"Kang, Moon, Hong, Lee, Cho and Kim (2002)" method:\n\n\t{CCT}'
)
print(colour.CCT_to_xy(CCT, method="Kang 2002"))
print(colour.temperature.CCT_to_xy_Kang2002(CCT))

print("\n")

message_box(
    f'Converting to "CIE xy" chromaticity coordinates from given "CCT" using '
    f'"CIE Illuminant D Series" method:\n\n\t{CCT}'
)
print(colour.CCT_to_xy(CCT, method="CIE Illuminant D Series"))
print(colour.temperature.CCT_to_xy_CIE_D(CCT))
