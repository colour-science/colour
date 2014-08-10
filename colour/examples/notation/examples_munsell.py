#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Munsell Renotation Sytem* computations.
"""

from numpy import array
from numpy import ravel

import colour
from colour.characterization.dataset.colour_checkers.chromaticity_coordinates import (
    COLORCHECKER_2005_DATA)

# Retrieving *RGB* *luminance* of given *RGB* components.
print(colour.get_RGB_luminance(array([56., 16., 100.]),
                               colour.sRGB_COLOURSPACE.primaries,
                               colour.sRGB_COLOURSPACE.whitepoint))

# Retrieving *Munsell* value and *Lightness* of given *xyY* components.
xyY = COLORCHECKER_2005_DATA[0][2:5]
Y = ravel(xyY)[2] * 100.
# Scaled *luminance* :math:`Y` reference:
print(Y)
# Retrieving *Munsell* value with *Priest et al.* 1920 method:
print(colour.munsell_value_priest1920(Y))
# Retrieving *Munsell* value with *Munsell, Sloan, and Godlove* 1933 method:
print(colour.munsell_value_munsell1933(Y))
# Retrieving *Munsell* value with *Moon and Spencer* 1943 method:
print(colour.munsell_value_moon1943(Y))
# Retrieving *Munsell* value with *Saunderson and Milner* 1944 method:
print(colour.munsell_value_saunderson1944(Y))
# Retrieving *Munsell* value with *Ladd and Pinney* 1955 method:
print(colour.munsell_value_ladd1955(Y))
# Retrieving *Munsell* value with *McCamy* 1987 method:
print(colour.munsell_value_mccamy1987(Y))
# Retrieving *Munsell* value with *ASTM D1535-08e1* 2008 method.
print(colour.munsell_value_ASTM_D1535_08(Y))
# Retrieving *Munsell* value using the wrapper:
print(colour.get_munsell_value(Y))
print(colour.get_munsell_value(Y, method="Munsell Value Saunderson 1944"))

# Converting from *CIE xyY* to *Munsell Colour*:
print(colour.xyY_to_munsell_colour(array([0.38736945, 0.35751656, 0.59362])))
# Converting from *CIE xyY* to *Munsell Colour*:
print(colour.munsell_colour_to_xyY("4.2YR 8.1/5.3"))
print(colour.munsell_colour_to_xyY("N8.9"))
