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
message_box(('Converting to "CCT" and "Duv" from given "CIE UCS" colourspace '
             '"uv" chromaticity coordinates using "Yoshi Ohno (2013)" '
             'method:\n'
             '\n\t{0}'.format(uv)))
print(colour.uv_to_CCT_ohno2013(uv, cmfs=cmfs))
print(colour.uv_to_CCT(uv, cmfs=cmfs))

print('\n')

message_box('Faster computation with 3 iterations but a lot less precise.')
print(colour.uv_to_CCT_ohno2013(uv, cmfs=cmfs, iterations=3))

print('\n')

message_box(('Converting to "CCT" and "Duv" from given "CIE UCS" colourspace '
             '"uv" chromaticity coordinates using "Robertson (1968)" method:\n'
             '\n\t{0}'.format(uv)))
print(colour.uv_to_CCT_robertson1968(uv))
print(colour.uv_to_CCT(uv, method='Robertson 1968'))

print('\n')

CCT, Duv = 6503.4925414981535, 0.0032059787171144823
message_box(('Converting to "CIE UCS" colourspace "uv" chromaticity '
             'coordinates from given "CCT" and "Duv" using '
             '"Yoshi Ohno (2013)" method:\n'
             '\n\t({0}, {1})'.format(CCT, Duv)))
print(colour.CCT_to_uv_ohno2013(CCT, Duv, cmfs=cmfs))
print(colour.CCT_to_uv(CCT, Duv, cmfs=cmfs))

print('\n')

message_box(('Converting to "CIE UCS" colourspace "uv" chromaticity '
             'coordinates from given "CCT" and "Duv" using "Robertson (1968)" '
             'method:\n'
             '\n\t({0}, {1})'.format(CCT, Duv)))
print(colour.CCT_to_uv_robertson1968(CCT, Duv))
print(colour.CCT_to_uv(CCT, Duv, method='Robertson 1968'))

print('\n')

xy = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
message_box(('Converting to "CCT" from given "xy" chromaticity coordinates '
             'using "McCamy (1992)" method:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_CCT_mccamy1992(xy))
print(colour.xy_to_CCT(xy, method='McCamy 1992'))

print('\n')

message_box(('Converting to "CCT" from given "xy" chromaticity coordinates '
             'using "Hernandez-Andres, Lee & Romero (1999)" method:\n'
             '\n\t{0}'.format(xy)))
print(colour.xy_to_CCT_hernandez1999(xy))
print(colour.xy_to_CCT(xy, method='Hernandez 1999'))

print('\n')

CCT = 6503.4925414981535
message_box(('Converting to "xy" chromaticity coordinates from given "CCT" '
             'using "Kang, Moon, Hong, Lee, Cho and Kim (2002)" method:\n'
             '\n\t{0}'.format(CCT)))
print(colour.CCT_to_xy_kang2002(CCT))
print(colour.CCT_to_xy(CCT, method="Kang 2002"))

print('\n')

message_box(('Converting to "xy" chromaticity coordinates from given "CCT" '
             'using "CIE Illuminant D Series" method:\n'
             '\n\t{0}'.format(CCT)))
print(colour.CCT_to_xy_illuminant_D(CCT))
print(colour.CCT_to_xy(CCT, method="CIE Illuminant D Series"))
