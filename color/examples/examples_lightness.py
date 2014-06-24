#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *luminance*, *Munsell value* and *Lightness* related examples.
"""

from numpy import matrix
from numpy import ravel

import color
import color.data.color_checkers.chromaticity_coordinates


# Retrieving *luminance* of given *RGB* components.
print color.get_luminance(matrix([56., 16., 100.]).reshape((3, 1)),
                          color.sRGB_COLORSPACE.primaries,
                          color.sRGB_COLORSPACE.whitepoint)

# Retrieving *Munsell value* and *Lightness* of given *xyY* components.
xyY = color.data.color_checkers.chromaticity_coordinates.COLORCHECKER_2005_DATA[0][2:5]
Y = ravel(xyY)[2] * 100.
# Scaled *luminance* *Y* reference:
print Y
# Retrieving *Munsell value* with *1920* method:
print color.munsell_value_1920(Y)
# Retrieving *Munsell value* with *1933* method:
print color.munsell_value_1933(Y)
# Retrieving *Munsell value* with *1943* method:
print color.munsell_value_1943(Y)
# Retrieving *Munsell value* with *1944* method:
print color.munsell_value_1944(Y)
# Retrieving *Munsell value* with *1955* method:
print color.munsell_value_1955(Y)
# Retrieving *Munsell value* using the wrapper:
print color.get_munsell_value(Y)
print color.get_munsell_value(Y, method="Munsell Value 1944")
# Retrieving *Lightness* *CIE Lab* reference:
print ravel(color.XYZ_to_Lab(color.xyY_to_XYZ(xyY)))[0]
# Retrieving *Lightness* with *1958* method:
print color.lightness_1958(Y)
# Retrieving *Lightness* with *1964* method:
print color.lightness_1964(Y)
# Retrieving *Lightness* with *1976* method:
print color.lightness_1976(Y)
# Retrieving *Lightness* using the wrapper:
print color.get_lightness(Y)
print color.get_lightness(Y, method="Lightness 1964")
