# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package *interpolation* related examples.
"""

import color
from color.implementations.matplotlib.plots import multi_spectral_power_distribution_plot


# Comparing *CIE* *Sprague* and *cubic spline* recommended interpolation methods.
uniform_spd_data = {340: 0.0000,
                    360: 0.0000,
                    380: 0.0000,
                    400: 0.0641,
                    420: 0.0645,
                    440: 0.0562,
                    460: 0.0537,
                    480: 0.0559,
                    500: 0.0651,
                    520: 0.0705,
                    540: 0.0772,
                    560: 0.0870,
                    580: 0.1128,
                    600: 0.1360,
                    620: 0.1511,
                    640: 0.1688,
                    660: 0.1996,
                    680: 0.2397,
                    700: 0.2852,
                    720: 0.0000,
                    740: 0.0000,
                    760: 0.0000,
                    780: 0.0000,
                    800: 0.0000,
                    820: 0.0000}

non_uniform_spd_data = {340: 0.0000,
                        361: 0.0000,
                        380: 0.0000,
                        400: 0.0641,
                        420: 0.0645,
                        440: 0.0562,
                        460: 0.0537,
                        480: 0.0559,
                        500: 0.0651,
                        520: 0.0705,
                        540: 0.0772,
                        560: 0.0870,
                        580: 0.1128,
                        600: 0.1360,
                        620: 0.1511,
                        640: 0.1688,
                        660: 0.1996,
                        680: 0.2397,
                        700: 0.2852,
                        720: 0.0000,
                        740: 0.0000,
                        760: 0.0000,
                        780: 0.0000,
                        800: 0.0000,
                        820: 0.0000}

base_spd = color.SpectralPowerDistribution("Reference", uniform_spd_data)
uniform_interpolated_spd = color.SpectralPowerDistribution("Uniform - Sprague Interpolation",
                                                           uniform_spd_data)
non_uniform_interpolated_spd = color.SpectralPowerDistribution("Non Uniform - Cubic Spline Interpolation",
                                                               non_uniform_spd_data)

uniform_interpolated_spd.interpolate(steps=1)
non_uniform_interpolated_spd.interpolate(steps=1)

multi_spectral_power_distribution_plot([base_spd, uniform_interpolated_spd, non_uniform_interpolated_spd])
