# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package *Munsell Renotation Sytem* related examples.
"""

from numpy import array
from numpy import ravel

import colour
import colour.dataset.colour_checkers.chromaticity_coordinates


# Retrieving *RGB* *luminance* of given *RGB* components.
print colour.get_RGB_luminance(array([56., 16., 100.]),
                               colour.sRGB_COLOURSPACE.primaries,
                               colour.sRGB_COLOURSPACE.whitepoint)

# Retrieving *Munsell value* and *Lightness* of given *xyY* components.
xyY = colour.dataset.colour_checkers.chromaticity_coordinates.COLORCHECKER_2005_DATA[0][2:5]
Y = ravel(xyY)[2] * 100.
# Scaled *luminance* *Y* reference:
print Y
# Retrieving *Munsell value* with *1920* method:
print colour.munsell_value_priest1920(Y)
# Retrieving *Munsell value* with *1933* method:
print colour.munsell_value_munsell1933(Y)
# Retrieving *Munsell value* with *1943* method:
print colour.munsell_value_moon1943(Y)
# Retrieving *Munsell value* with *1944* method:
print colour.munsell_value_saunderson1944(Y)
# Retrieving *Munsell value* with *1955* method:
print colour.munsell_value_ladd1955(Y)
# Retrieving *Munsell value* using the wrapper:
print colour.get_munsell_value(Y)
print colour.get_munsell_value(Y, method="Munsell Value Saunderson 1944")