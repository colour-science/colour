#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases correlated colour temperature computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('Correlated Colour Temperature Computations')

cmfs = colour.CMFS['CIE 1931 2 Degree Standard Observer']
illuminant = colour.ILLUMINANTS_RELATIVE_SPDS['D65']
xy = colour.XYZ_to_xy(colour.spectral_to_XYZ(illuminant, cmfs))
uv = colour.UCS_to_uv(colour.XYZ_to_UCS(colour.xy_to_XYZ(xy)))
message_box(('Converting to "CCT" and "D_uv" from given "CIE UCS" colourspace '
             '"uv" chromaticity coordinates using "Ohno (2013)" method:\n'
             '\n\t{0}'.format(uv)))
print(colour.uv_to_CCT_Ohno2013(uv, cmfs=cmfs))
print(colour.uv_to_CCT(uv, cmfs=cmfs))

print('\n')

message_box('Faster computation with 3 iterations but a lot less precise.')
print(colour.uv_to_CCT_Ohno2013(uv, cmfs=cmfs, iterations=3))

print('\n')

message_box(('Converting to "CCT" and "D_uv" from given "CIE UCS" colourspace '
             '"uv" chromaticity coordinates using "Robertson (1968)" method:\n'
             '\n\t{0}'.format(uv)))
print(colour.uv_to_CCT_Robertson1968(uv))
print(colour.uv_to_CCT(uv, method='Robertson 1968'))

print('\n')

CCT, D_uv = 6503.49254150, 0.00320598
message_box(('Converting to "CIE UCS" colourspace "uv" chromaticity '
             'coordinates from given "CCT" and "D_uv" using '
             '"Ohno (2013)" method:\n'
             '\n\t({0}, {1})'.format(CCT, D_uv)))
print(colour.CCT_to_uv_Ohno2013(CCT, D_uv, cmfs=cmfs))
print(colour.CCT_to_uv(CCT, D_uv=D_uv, cmfs=cmfs))

print('\n')

message_box(('Converting to "CIE UCS" colourspace "uv" chromaticity '
             'coordinates from given "CCT" and "D_uv" using '
             '"Robertson (1968)" method:\n'
             '\n\t({0}, {1})'.format(CCT, D_uv)))
print(colour.CCT_to_uv_Robertson1968(CCT, D_uv))
print(colour.CCT_to_uv(CCT, D_uv=D_uv, method='Robertson 1968'))

print('\n')

message_box(('Converting to "CIE UCS" colourspace "uv" chromaticity '
             'coordinates from given "CCT" using "Krystek (1985)" method:\n'
             '\n\t({0})'.format(CCT)))
print(colour.CCT_to_uv_Krystek1985(CCT))
print(colour.CCT_to_uv(CCT, method='Krystek 1985'))

print('\n')

xy = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
message_box(('Converting to "CCT" from given "xy" chromaticity coordinates '
             'using "McCamy (1992)" method:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_CCT_McCamy1992(xy))
print(colour.xy_to_CCT(xy, method='McCamy 1992'))

print('\n')

message_box(('Converting to "CCT" from given "xy" chromaticity coordinates '
             'using "Hernandez-Andres, Lee and Romero (1999)" method:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_CCT_Hernandez1999(xy))
print(colour.xy_to_CCT(xy, method='Hernandez 1999'))

print('\n')

CCT = 6503.49254150
message_box(('Converting to "xy" chromaticity coordinates from given "CCT" '
             'using "Kang, Moon, Hong, Lee, Cho and Kim (2002)" method:\n'
             '\n\t{0}'.format(CCT)))
print(colour.CCT_to_xy_Kang2002(CCT))
print(colour.CCT_to_xy(CCT, method="Kang 2002"))

print('\n')

message_box(('Converting to "xy" chromaticity coordinates from given "CCT" '
             'using "CIE Illuminant D Series" method:\n'
             '\n\t{0}'.format(CCT)))
print(colour.CCT_to_xy_CIE_D(CCT))
print(colour.CCT_to_xy(CCT, method="CIE Illuminant D Series"))
