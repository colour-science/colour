# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *interpolation* related examples.
"""

import pylab

import color
import color.implementations.matplotlib.plots


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

non_uniform_spd_data = {340.1: 0.0000,
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
                        820.9: 0.0000}

base_spd = color.SpectralPowerDistribution("Reference", uniform_spd_data)
uniform_interpolated_spd = color.SpectralPowerDistribution("Uniform - Sprague Interpolation",
                                                           uniform_spd_data)
non_uniform_interpolated_spd = color.SpectralPowerDistribution("Non Uniform - Cubic Spline Interpolation",
                                                               non_uniform_spd_data)

uniform_interpolated_spd.interpolate(steps=1)
non_uniform_interpolated_spd.interpolate(steps=1)

start, end, steps = base_spd.shape
x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []

pylab.plot(base_spd.wavelengths,
           base_spd.values,
           "ro-",
           label=base_spd.name,
           linewidth=2.)
pylab.plot(uniform_interpolated_spd.wavelengths,
           uniform_interpolated_spd.values,
           label=uniform_interpolated_spd.name,
           linewidth=2.)
pylab.plot(non_uniform_interpolated_spd.wavelengths,
           non_uniform_interpolated_spd.values,
           label=non_uniform_interpolated_spd.name,
           linewidth=2.)

x_limit_min.append(start)
x_limit_max.append(end)
y_limit_min.append(min(base_spd.values))
y_limit_max.append(max(base_spd.values))

settings = {"x_label": u"Wavelength λ (nm)",
            "y_label": "Spectral Power Distribution",
            "x_tighten": True,
            "legend": True,
            "legend_location": "upper left",
            "x_ticker": True,
            "y_ticker": True,
            "limits": [min(x_limit_min), max(x_limit_max), min(y_limit_min), max(y_limit_max)]}

color.implementations.matplotlib.plots.bounding_box(**settings)
color.implementations.matplotlib.plots.aspect(**settings)
color.implementations.matplotlib.plots.display(**settings)

# Comparing interpolation methods on problematic case.
boron_spd_data = {365.559: 33.028396,
                  366.01: 33.032729,
                  369.23: 19.178634,
                  369.534: 21.150628,
                  378.841: 21.150628,
                  394.628: 24.51399,
                  395.037: 24.514172,
                  399.024: 21.150628,
                  400.007: 228.47965,
                  412.193: 18.678179,
                  414.725: 33.028396,
                  419.479: 17.866487,
                  419.777: 250.30083,
                  424.298: 30.107049,
                  424.361: 30.107534,
                  427.273: 20.624137,
                  429.571: 21.150628,
                  436.147: 233.11958,
                  436.611: 233.12144,
                  443.15: 24.857254,
                  445.944: 250.54088,
                  447.21: 17.852553,
                  447.25: 32.627653,
                  447.285: 17.852994,
                  448.705: 30.269855,
                  449.08: 202.80222,
                  449.773: 30.276908,
                  450.481: 30.276831,
                  459.73: 232.65454,
                  461.113: 21.268987,
                  463.249: 30.269703,
                  464.713: 250.65717,
                  465.815: 250.66419,
                  468.5: 253.3251,
                  471.611: 24.514172,
                  471.97: 250.69717,
                  477.33: 250.65717,
                  478.42: 18.678179,
                  481.277: 250.54088,
                  491.743: 30.107049,
                  491.843: 30.107534,
                  494.037: 19.178634,
                  494.46: 330.7779940153,
                  498.866: 32.945545,
                  512.579: 21.582341,
                  515.785: 33.028396,
                  516.596: 33.032203,
                  522.686: 33.028396,
                  526.312: 21.685248,
                  529.14: 32.945545,
                  534.765: 20.821324,
                  539.317: 21.150628,
                  550.456: 5.933527,
                  556.318: 5.933527,
                  563.306: 4.96428874,
                  564.432: 5.933527,
                  576.194: 5.933527,
                  581.84: 8.991693,
                  582.107: 8.993138,
                  582.233: 8.993138,
                  594.263: 5.9334878,
                  594.272: 5.933527,
                  601.346: 21.268987,
                  608.044: 15.82797,
                  612.502: 218.65175,
                  612.79: 218.65175,
                  614.892: 21.582341,
                  624.457: 5.9334878,
                  624.466: 5.933527,
                  628.547: 19.178634,
                  634.915: 21.685248,
                  635.681: 21.687549,
                  657.13: 22.53607,
                  671.765: 21.765054,
                  678.612: 21.150628,
                  703.02: 16.089904,
                  703.19: 16.089904,
                  703.254: 16.089904,
                  783.525: 22.34219,
                  784.141: 22.34219,
                  821.183: 5.9334878,
                  821.204: 5.933527}

boron_spd = color.SpectralPowerDistribution("Boron", boron_spd_data)
cubic_spline_interpolated_boron_spd = color.SpectralPowerDistribution("Non Uniform - Cubic Spline Interpolation",
                                                                boron_spd_data)

cubic_spline_interpolated_boron_spd.interpolate(steps=1)

start, end, steps = boron_spd.shape
x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []

pylab.plot(boron_spd.wavelengths,
           boron_spd.values,
           "ro-",
           label=boron_spd.name,
           linewidth=2.)
pylab.plot(cubic_spline_interpolated_boron_spd.wavelengths,
           cubic_spline_interpolated_boron_spd.values,
           label=cubic_spline_interpolated_boron_spd.name,
           linewidth=2.)

x_limit_min.append(start)
x_limit_max.append(end)
y_limit_min.append(min(base_spd.values))
y_limit_max.append(max(base_spd.values))

settings = {"x_label": u"Wavelength λ (nm)",
            "y_label": "Spectral Power Distribution",
            "x_tighten": True,
            "legend": True,
            "legend_location": "upper left",
            "x_ticker": True,
            "y_ticker": True,
            "limits": [min(x_limit_min), max(x_limit_max), min(y_limit_min), max(y_limit_max)]}

color.implementations.matplotlib.plots.bounding_box(**settings)
color.implementations.matplotlib.plots.aspect(**settings)
color.implementations.matplotlib.plots.display(**settings)
