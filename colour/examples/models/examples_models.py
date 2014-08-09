#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package *colourspaces* related examples.
"""

from numpy import array
import colour

# Displaying explicit *A* illuminant chromaticity coordinates under
# *CIE 1931 2 Degree Standard Observer*.
print(colour.ILLUMINANTS.get("CIE 1931 2 Degree Standard Observer").get("A"))

# From *CIE XYZ* colourspace to *CIE xyY* colourspace.
print(colour.XYZ_to_xyY(array([[0.1180583421], [0.1034], [0.0515089229]])))

# Any definitions accepting 3 x 1 matrices will accept a tuple / list input.
print(colour.XYZ_to_xyY([0.1180583421, 0.1034, 0.0515089229]))

# Default reference illuminant in case X == Y == Z == 0 is *D50*.
print(colour.XYZ_to_xyY(array([[0], [0], [0]])))

# Using an alternative illuminant.
print(colour.XYZ_to_xyY(
    array([[0], [0], [0]]),
    colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D60"]))

# From *CIE xyY* colourspace to *CIE XYZ* colourspace.
print(colour.xyY_to_XYZ(array([[0.4325], [0.3788], [10.34]])))

# From chromaticity coordinates to *CIE XYZ* colourspace.
print(colour.xy_to_XYZ((0.25, 0.25)))

# From *CIE XYZ* colourspace to chromaticity coordinates.
print(colour.XYZ_to_xy(array([[0.97137399], [1.], [1.04462134]])))

# From *CIE XYZ* colourspace to *RGB* colourspace.
# From *CIE XYZ* colourspace to *sRGB* colourspace.
print(colour.XYZ_to_RGB(
    array([0.1180583421, 0.1034, 0.0515089229]),
    colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
    colour.sRGB_COLOURSPACE.whitepoint,
    "Bradford",
    colour.sRGB_COLOURSPACE.from_XYZ,
    colour.sRGB_COLOURSPACE.transfer_function))

# From *RGB* colourspace to *CIE XYZ* colourspace.
# From *sRGB* colourspace to *CIE XYZ* colourspace.
print(colour.RGB_to_XYZ(
    array([[3.40552203], [2.48159742], [2.11932818]]),
    colour.sRGB_COLOURSPACE.whitepoint,
    colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
    "Bradford",
    colour.sRGB_COLOURSPACE.to_XYZ,
    colour.sRGB_COLOURSPACE.inverse_transfer_function))

# From *CIE xyY* colourspace to *RGB* colourspace.
# From *CIE xyY* colourspace to *sRGB* colourspace.
print(colour.xyY_to_RGB(
    array([[0.4316], [0.3777], [0.1008]]),
    colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
    colour.sRGB_COLOURSPACE.whitepoint,
    "Bradford",
    colour.sRGB_COLOURSPACE.from_XYZ,
    colour.sRGB_COLOURSPACE.transfer_function))

# From *RGB* colourspace to *CIE xyY* colourspace.
# From *sRGB* colourspace to *CIE xyY* colourspace.
print(colour.RGB_to_xyY(
    array([[3.40552203], [2.48159742], [2.11932818]]),
    colour.sRGB_COLOURSPACE.whitepoint,
    colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
    "Bradford",
    colour.sRGB_COLOURSPACE.to_XYZ,
    colour.sRGB_COLOURSPACE.inverse_transfer_function))

# From *CIE XYZ* colourspace to *CIE UCS* colourspace.
print(colour.XYZ_to_UCS(array([[0.92193107], [1.], [1.03744246]])))

# From *CIE UCS* colourspace to *CIE XYZ* colourspace.
print(colour.UCS_to_XYZ(array([[0.61462071], [1.], [1.55775569]])))

# From *CIE UCS* colourspace to *uv* chromaticity coordinates.
print(colour.UCS_to_uv(array([[0.61462071], [1.], [1.55775569]])))

# From *CIE UCS* colourspace *uv* chromaticity coordinates to *xy* chromaticity
# coordinates.
print(colour.UCS_uv_to_xy((0.19374142046952561, 0.31522110680182841)))

# From *CIE XYZ* colourspace to *CIE UVW* colourspace.
print(colour.XYZ_to_UVW(array([[0.92193107], [1.], [1.03744246]])))

# From *CIE XYZ* colourspace to *CIE Luv* colourspace.
print(colour.XYZ_to_Luv(array([[0.92193107], [1.], [1.03744246]])))

# From *CIE Luv* colourspace to *CIE XYZ* colourspace.
print(colour.Luv_to_XYZ(array([[100.], [-20.04304247], [-19.81676035]])))

# From *CIE Luv* colourspace to *uv* chromaticity coordinates.
print(colour.Luv_to_uv(array([[100.], [-20.04304247], [-19.81676035]])))

# From *CIE Luv* colourspace *uv* chromaticity coordinates to *xy* chromaticity
# coordinates.
print(colour.Luv_uv_to_xy((0.19374142100850045, 0.47283165896209456)))

# From *CIE Luv* colourspace to *CIE LCHuv* colourspace.
print(colour.Luv_to_LCHuv(array([[100.], [-20.04304247], [-19.81676035]])))

# From *CIE LCHuv* colourspace to *CIE Luv* colourspace.
print(colour.LCHuv_to_Luv(array([[100.], [28.18559104], [224.6747382]])))

# From *CIE XYZ* colourspace to *CIE Lab* colourspace.
print(colour.XYZ_to_Lab(array([[0.92193107], [1.], [1.03744246]])))

# From *CIE Lab* colourspace to *CIE XYZ* colourspace.
print(colour.Lab_to_XYZ(array([[100.], [-7.41787844], [-15.85742105]])))

# From *CIE Lab* colourspace to *CIE LCHab* colourspace.
print(colour.Lab_to_LCHab(array([[100.], [-7.41787844], [-15.85742105]])))

# From *CIE LCHab* colourspace to *CIE Lab* colourspace.
print(colour.LCHab_to_Lab(array([[100.], [17.50664796], [244.93046842]])))
