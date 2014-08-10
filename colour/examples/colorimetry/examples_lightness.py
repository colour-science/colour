#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Lightness* computations.
"""

from numpy import ravel

import colour
from colour.characterization.dataset.colour_checkers.chromaticity_coordinates import (
    COLORCHECKER_2005_DATA)

# Retrieving *Lightness* of given *xyY* components.
xyY = COLORCHECKER_2005_DATA[0][2:5]
Y = ravel(xyY)[2] * 100.
# Scaled *luminance* :math:`Y` reference:
print(Y)
# Retrieving *Lightness* *CIE Lab* reference:
print(ravel(colour.XYZ_to_Lab(colour.xyY_to_XYZ(xyY)))[0])
# Retrieving *Lightness* with *Glasser et al.* 1958 method:
print(colour.lightness_glasser1958(Y))
# Retrieving *Lightness* with *Wyszecki* 1964 method:
print(colour.lightness_wyszecki1964(Y))
# Retrieving *Lightness* with *1976* method:
print(colour.lightness_1976(Y))
# Retrieving *Lightness* using the wrapper:
print(colour.get_lightness(Y))
print(colour.get_lightness(Y, method="Lightness Wyszecki 1964"))
