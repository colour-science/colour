#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *ColorChecker* related examples.
"""

from numpy import matrix
from numpy import ravel
import color

# Displaying :attr:`color.color_checkers.COLORCHECKERS` data.
name, data, illuminant = color.COLORCHECKERS["ColorChecker 2005"]
for index, name, x, y, Y in data:
    print(index, name, x, y, Y)

# Converting *ColorChecker 2005* color checker *CIE xyY* colorspace values to *sRGB* colorspace.
for index, name, x, y, Y in data:
    RGB = color.xyY_to_RGB(matrix([[x], [y], [Y]]),
                           illuminant,
                           color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D65"],
                           "Bradford",
                           color.sRGB_COLORSPACE.from_XYZ,
                           color.sRGB_COLORSPACE.transfer_function)

    RGB = map(lambda x: int(round(x * 255)) if x >= 0 else 0, ravel(RGB))
    print("'{0}': {1}".format(name, RGB))
