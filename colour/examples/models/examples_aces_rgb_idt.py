#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *ACES RGB* colourspace *Input Device Transform* related computations.
"""

import colour

# Calculating *ACES RGB* colourspace relative exposure values for some colour
# checker spectral power distributions.
print(colour.spectral_to_aces_relative_exposure_values(
    colour.COLOURCHECKERS_SPDS['ColorChecker N Ohta']['dark skin']))
print(colour.spectral_to_aces_relative_exposure_values(
    colour.COLOURCHECKERS_SPDS['ColorChecker N Ohta']['blue sky']))

# Calculating *ACES RGB* colourspace relative exposure values for various ideal
# reflectors.
wavelengths = colour.ACES_RICD.wavelengths
gray_reflector = colour.SpectralPowerDistribution(
    '18%', dict(zip(wavelengths, [0.18] * len(wavelengths))))
print(repr(colour.spectral_to_aces_relative_exposure_values(gray_reflector)))

perfect_reflector = colour.SpectralPowerDistribution(
    '100%', dict(zip(wavelengths, [1.] * len(wavelengths))))
print(colour.spectral_to_aces_relative_exposure_values(perfect_reflector))
