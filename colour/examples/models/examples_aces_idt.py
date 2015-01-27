#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Academy Color Encoding System* *Input Device Transform* related
computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"ACES" "Input Device Transform" Computations')

message_box(('Computing "ACES" relative exposure '
             'values for some colour rendition chart spectral power '
             'distributions:\n'
             '\n\t("dark skin", \n\t"blue sky")'))
print(colour.spectral_to_aces_relative_exposure_values(
    colour.COLOURCHECKERS_SPDS['ColorChecker N Ohta']['dark skin']))
print(colour.spectral_to_aces_relative_exposure_values(
    colour.COLOURCHECKERS_SPDS['ColorChecker N Ohta']['blue sky']))

print('\n')

message_box(('Computing "ACES" relative exposure values for various ideal '
             'reflectors:\n'
             '\n\t("18%", \n\t"100%")'))
wavelengths = colour.ACES_RICD.wavelengths
gray_reflector = colour.SpectralPowerDistribution(
    '18%', dict(zip(wavelengths, [0.18] * len(wavelengths))))
print(repr(colour.spectral_to_aces_relative_exposure_values(gray_reflector)))

perfect_reflector = colour.SpectralPowerDistribution(
    '100%', dict(zip(wavelengths, [1.] * len(wavelengths))))
print(colour.spectral_to_aces_relative_exposure_values(perfect_reflector))
