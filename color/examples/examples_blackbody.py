#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *blackbody* related examples.
"""

import numpy
import color

# Converting temperature to *CIE XYZ*.
cmfs = color.STANDARD_OBSERVERS_XYZ_CMFS.get("CIE 1931 2 Degree Standard Observer")
spd = color.blackbody_spectral_power_distribution(5000, *cmfs.shape)
XYZ = color.spectral_to_XYZ(spd, cmfs)
XYZ *= 1. / numpy.max(XYZ)
print(XYZ)
