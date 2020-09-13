# -*- coding: utf-8 -*-
"""
Showcases reflectance recovery computations using *Jakob et al. (2019)* method.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('"Jakob et al. (2019)" - Reflectance Recovery Computations')

illuminant = colour.SDS_ILLUMINANTS['D65']

XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
message_box('Recovering reflectance using "Jakob et al. (2019)" method from '
            'given "XYZ" tristimulus values:\n'
            '\n\tXYZ: {0}'.format(XYZ))
sd = colour.XYZ_to_sd(XYZ, method='Jakob 2019')
print(sd)
print(colour.recovery.XYZ_to_sd_Jakob2019(XYZ))
print(colour.sd_to_XYZ(sd, illuminant=illuminant) / 100)

print('\n')

message_box('Generating a LUT according to "Jakob et al. (2019)" method '
            'for "sRGB" colourspace:')
LUT = colour.recovery.LUT3D_Jakob2019()
LUT.generate(colour.models.RGB_COLOURSPACE_sRGB, size=5)
RGB = np.array([0.70573936, 0.19248266, 0.22354169])
print(LUT.RGB_to_sd(RGB))
