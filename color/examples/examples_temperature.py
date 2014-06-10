#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package correlated color temperature related examples.
"""

from numpy import matrix
import color

# From *uv* chromaticity coordinates to correlated color temperature.
# Default to *Yoshi Ohno* implementation.
cmfs = color.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS["Standard CIE 1931 2 Degree Observer"]
illuminant = color.ILLUMINANTS_RELATIVE_SPD["D65"].zeros(*cmfs.shape)
xy = color.XYZ_to_xy(color.spectral_to_XYZ(illuminant, cmfs))
uv = color.UCS_to_uv(color.XYZ_to_UCS(color.xy_to_XYZ(xy)))
print(color.uv_to_cct(uv, cmfs=cmfs))

# Faster but less precise version.
print(color.uv_to_cct(uv, cmfs=cmfs, iterations=3))

# *Wyszecki & Roberston* calculation method.
print(color.uv_to_cct(uv, method="Wyszecki Robertson", cmfs=cmfs, iterations=3))

# From correlated color temperature to *uv* chromaticity coordinates.
print(color.cct_to_uv(6503.4925414981535, 0.0032059787171144823, cmfs=cmfs))

# *Wyszecki & Roberston* calculation method.
print(color.cct_to_uv(6503.4925414981535, 0.0032059787171144823, method="Wyszecki Robertson", cmfs=cmfs))
