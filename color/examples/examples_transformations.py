#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package color *transformations* related examples.
"""

from numpy import matrix
import color

# Displaying explicit *A* illuminant chromaticity coordinates under *Standard CIE 1931 2 Degree Observer*.
print(color.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("A"))

# From *CIE XYZ* colorspace to *CIE xyY* colorspace.
print(color.XYZ_to_xyY(matrix([[11.80583421], [10.34], [5.15089229]])))

# Any definitions accepting 3 x 1 matrices will accept a tuple / list input.
print(color.XYZ_to_xyY([11.80583421, 10.34, 5.15089229]))

# Default reference illuminant in case X == Y == Z == 0 is *D50*.
print(color.XYZ_to_xyY(matrix([[0], [0], [0]])))

# Using an alternative illuminant.
print(color.XYZ_to_xyY(matrix([[0], [0], [0]]), color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D60"]))

# From *CIE xyY* colorspace to *CIE XYZ* colorspace.
print(color.xyY_to_XYZ(matrix([[0.4325], [0.3788], [10.34]])))

# From chromaticity coordinates to *CIE XYZ* colorspace.
print(color.xy_to_XYZ((0.25, 0.25)))

# From *CIE XYZ* colorspace to chromaticity coordinates.
print(color.XYZ_to_xy(matrix([[0.97137399], [1.], [1.04462134]])))

# From *CIE XYZ* colorspace to *RGB* colorspace.
# From *CIE XYZ* colorspace to *sRGB* colorspace.
print(color.XYZ_to_RGB(matrix([[11.51847498], [10.08], [5.08937252]]),
                       color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D50"],
                       color.sRGB_COLORSPACE.whitepoint,
                       "Bradford",
                       color.sRGB_COLORSPACE.from_XYZ,
                       color.sRGB_COLORSPACE.transfer_function))

# From *RGB* colorspace to *CIE XYZ* colorspace.
# From *sRGB* colorspace to *CIE XYZ* colorspace.
print(color.RGB_to_XYZ(matrix([[3.40552203], [2.48159742], [2.11932818]]),
                       color.sRGB_COLORSPACE.whitepoint,
                       color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D50"],
                       "Bradford",
                       color.sRGB_COLORSPACE.to_XYZ,
                       color.sRGB_COLORSPACE.inverse_transfer_function))

# From *CIE xyY* colorspace to *RGB* colorspace.
# From *CIE xyY* colorspace to *sRGB* colorspace.
print(color.xyY_to_RGB(matrix([[0.4316], [0.3777], [10.08]]),
                       color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D50"],
                       color.sRGB_COLORSPACE.whitepoint,
                       "Bradford",
                       color.sRGB_COLORSPACE.from_XYZ,
                       color.sRGB_COLORSPACE.transfer_function))

# From *RGB* colorspace to *CIE xyY* colorspace.
# From *sRGB* colorspace to *CIE xyY* colorspace.
print(color.RGB_to_xyY(matrix([[3.40552203], [2.48159742], [2.11932818]]),
                       color.sRGB_COLORSPACE.whitepoint,
                       color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D50"],
                       "Bradford",
                       color.sRGB_COLORSPACE.to_XYZ,
                       color.sRGB_COLORSPACE.inverse_transfer_function))

# From *CIE XYZ* colorspace to *CIE UVW* colorspace.
print(color.XYZ_to_UVW(matrix([[0.92193107], [1.], [1.03744246]])))

# From *CIE UVW* colorspace to *CIE XYZ* colorspace.
print(color.UVW_to_XYZ(matrix([[0.61462071], [1.], [1.55775569]])))

# From *CIE UVW* colorspace to *uv* chromaticity coordinates.
print(color.UVW_to_uv(matrix([[0.61462071], [1.], [1.55775569]])))

# From *CIE UVW* colorspace *uv* chromaticity coordinates to *xy* chromaticity coordinates.
print(color.UVW_uv_to_xy((0.19374142046952561, 0.31522110680182841)))

# From *CIE XYZ* colorspace to *CIE Luv* colorspace.
print(color.XYZ_to_Luv(matrix([[0.92193107], [1.], [1.03744246]])))

# From *CIE Luv* colorspace to *CIE XYZ* colorspace.
print(color.Luv_to_XYZ(matrix([[100.], [-20.04304247], [-19.81676035]])))

# From *CIE Luv* colorspace to *uv* chromaticity coordinates.
print(color.Luv_to_uv(matrix([[100.], [-20.04304247], [-19.81676035]])))

# From *CIE Luv* colorspace *uv* chromaticity coordinates to *xy* chromaticity coordinates.
print(color.Luv_uv_to_xy((0.19374142100850045, 0.47283165896209456)))

# From *CIE Luv* colorspace to *CIE LCHuv* colorspace.
print(color.Luv_to_LCHuv(matrix([[100.], [-20.04304247], [-19.81676035]])))

# From *CIE LCHuv* colorspace to *CIE Luv* colorspace.
print(color.LCHuv_to_Luv(matrix([[100.], [28.18559104], [224.6747382]])))

# From *CIE XYZ* colorspace to *CIE Lab* colorspace.
print(color.XYZ_to_Lab(matrix([[0.92193107], [1.], [1.03744246]])))

# From *CIE Lab* colorspace to *CIE XYZ* colorspace.
print(color.Lab_to_XYZ(matrix([[100.], [-7.41787844], [-15.85742105]])))

# From *CIE Lab* colorspace to *CIE LCHab* colorspace.
print(color.Lab_to_LCHab(matrix([[100.], [-7.41787844], [-15.85742105]])))

# From *CIE LCHab* colorspace to *CIE Lab* colorspace.
print(color.LCHab_to_Lab(matrix([[100.], [17.50664796], [244.93046842]])))
