# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package correlated colour temperature related examples.
"""

import colour

# From *uv* chromaticity coordinates to correlated colour temperature.
# Default to *Yoshi Ohno* implementation.
cmfs = colour.CMFS["CIE 1931 2 Degree Standard Observer"]
illuminant = colour.ILLUMINANTS_RELATIVE_SPDS["D65"]
xy = colour.XYZ_to_xy(colour.spectral_to_XYZ(illuminant, cmfs))
uv = colour.UCS_to_uv(colour.XYZ_to_UCS(colour.xy_to_XYZ(xy)))
print(colour.uv_to_CCT(uv, cmfs=cmfs))

# Faster but a lot less precise version.
print(colour.uv_to_CCT(uv, cmfs=cmfs, iterations=3))

# *Roberston* calculation method.
print(colour.uv_to_CCT(uv, method="Robertson", cmfs=cmfs, iterations=3))

# From correlated colour temperature to *uv* chromaticity coordinates.
print(colour.CCT_to_uv(6503.4925414981535, 0.0032059787171144823, cmfs=cmfs))

# *Roberston* calculation method.
print(colour.CCT_to_uv(6503.4925414981535, 0.0032059787171144823, method="Robertson", cmfs=cmfs))


# From *xy* chromaticity coordinates to correlated colour temperature.
# *McCamy* calculation method.
xy = colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
print(colour.xy_to_CCT_mccamy(xy))

# From *xy* chromaticity coordinates to correlated colour temperature.
# *Hernandez-Andres, Lee & Romero* calculation method.
print(colour.xy_to_CCT_lee(xy))

print colour.XYZ_to_xy(colour.spectral_to_XYZ(colour.blackbody_spectral_power_distribution(65000, *cmfs.shape), cmfs))

