# -*- coding: utf-8 -*-
"""
Showcases blackbody / planckian radiator computations.
"""

import colour
from colour.utilities import message_box

message_box('Blackbody / Planckian Radiator Computations')

message_box(('Computing the spectral distribution of a blackbody at '
             'temperature 5500 kelvin degrees and converting to "CIE XYZ" '
             'tristimulus values.'))
cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
blackbody_sd = colour.sd_blackbody(5500, cmfs.shape)
print(blackbody_sd)
XYZ = colour.sd_to_XYZ(blackbody_sd, cmfs)
print(XYZ)

print('\n')

message_box(('Computing the spectral radiance of a blackbody at wavelength '
             '500 nm and temperature 5500 kelvin degrees.'))
print(colour.colorimetry.blackbody_spectral_radiance(500 * 1e-9, 5500))
print(colour.colorimetry.planck_law(500 * 1e-9, 5500))
