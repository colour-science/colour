#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package *blackbody* related examples.
"""

import numpy
import color

# Converting temperature to *CIE XYZ*.
cmfs = color.STANDARD_OBSERVERS_XYZ_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")
spd = color.blackbodySpectralPowerDistribution(5000, *cmfs.shape)
XYZ = color.spectral_to_XYZ(spd, cmfs)
XYZ *= 1. / numpy.max(XYZ)
print(XYZ)
