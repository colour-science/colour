# -*- coding: utf-8 -*-
"""
Showcases *Academy Color Encoding System* *Input Device Transform* related
computations.
"""

import colour
from colour.utilities import message_box

message_box('"ACES" "Input Device Transform" Computations')

message_box(('Computing "ACES" relative exposure '
             'values for some colour rendition chart spectral '
             'distributions:\n'
             '\n\t("dark skin", \n\t"blue sky")'))
print(
    colour.sd_to_aces_relative_exposure_values(
        colour.COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']))
print(
    colour.sd_to_aces_relative_exposure_values(
        colour.COLOURCHECKERS_SDS['ColorChecker N Ohta']['blue sky']))

print('\n')

message_box(('Computing "ACES" relative exposure values for various ideal '
             'reflectors:\n'
             '\n\t("18%", \n\t"100%")'))
wavelengths = colour.models.ACES_RICD.wavelengths
gray_reflector = colour.SpectralDistribution(
    dict(zip(wavelengths, [0.18] * len(wavelengths))), name='18%')
print(repr(colour.sd_to_aces_relative_exposure_values(gray_reflector)))

perfect_reflector = colour.SpectralDistribution(
    dict(zip(wavelengths, [1.] * len(wavelengths))), name='100%')
print(colour.sd_to_aces_relative_exposure_values(perfect_reflector))
