#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *blackbody* related examples.
"""

import color

# Calculating *blackbody* spectral radiance.
color.planck_law(500 * 1e-9, 5500)

# Converting temperature to *CIE XYZ*.
cmfs = color.STANDARD_OBSERVERS_XYZ_CMFS.get("CIE 1931 2 Degree Standard Observer")
blackbody_spd = color.blackbody_spectral_power_distribution(5000, *cmfs.shape)
XYZ = color.spectral_to_XYZ(blackbody_spd, cmfs)
print(XYZ)
