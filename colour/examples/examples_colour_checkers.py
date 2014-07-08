# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package *ColourChecker* related examples.
"""

from numpy import matrix
from numpy import ravel
import colour

# Displaying :attr:`colour.colour_checkers.COLOURCHECKERS` data.
name, data, illuminant = colour.COLOURCHECKERS["ColorChecker 2005"]
for index, name, x, y, Y in data:
    print(index, name, x, y, Y)

# Converting *ColorChecker 2005* colour checker *CIE xyY* colourspace values to *sRGB* colourspace.
for index, name, x, y, Y in data:
    RGB = colour.xyY_to_RGB(matrix([[x], [y], [Y]]),
                           illuminant,
                           colour.ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
                           "Bradford",
                           colour.sRGB_COLOURSPACE.from_XYZ,
                           colour.sRGB_COLOURSPACE.transfer_function)

    RGB = map(lambda x: int(round(x * 255)) if x >= 0 else 0, ravel(RGB))
    print("'{0}': {1}".format(name, RGB))
